"""临时按钮模型的可逐步交互仿真核心；不修改正式 XML。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from experiment_small_normal_press import (
    build_experiment_xml, collision_pair_audit, joint_displacement,
)
from simulator import IndexSimulator
from button_press_state import ButtonPressState


class SimulationFault(RuntimeError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def _build_temporary_model(target_button=None):
    base = Path(__file__).resolve().parent
    root = base.parent
    out = root / 'artifacts/trajectory_check_2026-10-22'
    panel = base / 'simulation.xml'
    pointing = root / 'mobl_arms_index_pointing/simulation.xml'
    originals = [p.read_bytes() for p in (panel, pointing)]
    scan = json.loads((root / 'artifacts/trajectory_check_2026-10-21/'
                      'shoulder_elbow_reachability.json').read_text())
    single_button_baseline = target_button is None
    if single_button_baseline:
        target_button = scan['target_button']
    if type(target_button) is not int or target_button not in range(4):
        raise ValueError('target_button must be an integer in [0, 3]')
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', panel)
        sim.reset()
    try:
        source = sim.model
        initial = sim.data.qpos.copy()
        fixture = {n: float(initial[source.jnt_qposadr[source.joint(n).id]])
                   for n in ('head_rot', 'elv_angle', 'shoulder_elv', 'pro_sup')}
        pairs = [tuple(e['body'] for e in pair)
                 for pair in collision_pair_audit(source)['pairs']]
        button_name = f"button-{target_button}"
        with tempfile.TemporaryDirectory(prefix='full_slider_baseline_') as directory:
            xml = Path(directory) / 'baseline.xml'
            build_experiment_xml(panel, pointing, xml, fixture, pairs)
            tree = ET.parse(xml)
            body = tree.getroot().find(f"./worldbody/body[@name='{button_name}']")
            ET.SubElement(body, 'joint', name='probe_button_slide', type='slide',
                          axis='0 0 -1', limited='true', range='0 0.005',
                          stiffness='1000', damping='10', springref='0',
                          solreflimit='0.004 1')
            tree.write(xml, encoding='utf-8')
            model = mujoco.MjModel.from_xml_path(str(xml))
            model.opt.timestep = 0.002
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            data = mujoco.MjData(model)
            # A new joint shifts array addresses: copy full blocks by name.
            for jid in range(source.njnt):
                dest = int(model.jnt_qposadr[model.joint(source.joint(jid).name).id])
                start = int(source.jnt_qposadr[jid])
                end = int(source.jnt_qposadr[jid + 1]) if jid + 1 < source.njnt else source.nq
                data.qpos[dest:dest + end - start] = initial[start:end]
                np.testing.assert_array_equal(data.qpos[dest:dest + end - start], initial[start:end])
            finger, button = model.geom('hand_2distph').id, model.geom(button_name).id
            model.geom_margin[finger] = 0.0
            joints = [model.joint(n).id for n in ('shoulder_rot', 'elbow_flexion')]
            qadr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
            if single_button_baseline:
                data.qpos[qadr] = [scan['nearest_surface'][name + '_rad']
                                   for name in ('shoulder_rot', 'elbow_flexion')]
            else:
                multidof = json.loads((root / 'artifacts/trajectory_check_2026-11-08/'
                                       'button_reachability_multidof.json').read_text())
                values = multidof['argmin_joint_values_rad'][str(target_button)]
                for name, value in zip(multidof['joints'], values):
                    data.qpos[int(model.jnt_qposadr[model.joint(name).id])] = value
            mujoco.mj_forward(model, data)
            normal = data.geom_xmat[button].reshape(3, 3)[:, 2].copy()
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, model.site('fingertip').id)
            q_out = data.qpos[qadr] + joint_displacement(model, data, jac, normal, 0.006, dofs)
            if single_button_baseline:
                press_scan = json.loads((out / 'filtered_press_target_scan.json').read_text())
                pose = press_scan['selected_candidate_by_tangential_alignment_within_0_2_to_0_4mm_clearance']
                data.qpos[qadr] = np.radians([pose['shoulder_rot_deg'], pose['elbow_flexion_deg']])
                mujoco.mj_forward(model, data)
            mujoco.mj_jacSite(model, data, jac, None, model.site('fingertip').id)
            q_press = data.qpos[qadr] + joint_displacement(model, data, jac, normal, -0.0008, dofs)
            data.qpos[qadr] = q_out
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            slider = model.joint('probe_button_slide').id
            sq, sd = int(model.jnt_qposadr[slider]), int(model.jnt_dofadr[slider])
            gravity_force = -float(data.qfrc_bias[sd])
            model.qpos_spring[sq] = -gravity_force / 1000.0
            mujoco.mj_forward(model, data)
            motors = [model.actuator(n).id for n in ('press_shoulder_motor', 'press_elbow_motor')]
            return dict(model=model, initial_data=data, finger=finger, button=button,
                        qadr=qadr, dofs=dofs, sq=sq, sd=sd, motors=motors,
                        q_out=q_out, q_press=q_press, target_button=target_button)
    finally:
        sim.close()
        assert [p.read_bytes() for p in (panel, pointing)] == originals


class ButtonSliderSimulation:
    """Temporary two-joint PD simulation, one 2ms physics step per call.

    Targets are shoulder/elbow angles (rad) and angular velocities (rad/s).
    This is a simulation core; reward and episode termination are not defined.
    """

    def __init__(self, target_button=None):
        """Use the training baseline by default, or probe an explicit four-DOF candidate."""
        self.__dict__.update(_build_temporary_model(target_button=target_button))
        self.task = ButtonPressState()
        self._initialized = False

    def reset(self, initial_offset_rad=None):
        # Restore all MjData, including solver history, rather than qpos alone.
        self.data = copy.copy(self.initial_data)
        offset = (np.zeros(2, dtype=float) if initial_offset_rad is None
                  else np.asarray(initial_offset_rad, dtype=float))
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError('Initial pose offset must be a finite vector of shape (2,)')
        self.data.qpos[self.qadr] += offset
        self.data.qvel[:] = 0
        joint_ranges = self.model.jnt_range[
            [self.model.joint('shoulder_rot').id, self.model.joint('elbow_flexion').id]]
        if np.any(self.data.qpos[self.qadr] < joint_ranges[:, 0]) or np.any(
                self.data.qpos[self.qadr] > joint_ranges[:, 1]):
            raise ValueError('Initial pose offset moves a joint outside its model range')
        mujoco.mj_forward(self.model, self.data)
        self.task.reset()
        self.step_count = 0
        self.saturation_count = 0
        self.last_saturated_motor_channels = 0
        self.last_warnings = []
        self.q_target = self.q_out.copy()
        self._initialized = True
        return self._observe()

    def step(self, target_rad, target_velocity_rad_s=None):
        if not self._initialized:
            raise RuntimeError('Call reset() before step()')
        target = np.asarray(target_rad, dtype=float)
        velocity = np.zeros(2) if target_velocity_rad_s is None else np.asarray(target_velocity_rad_s, dtype=float)
        if target.shape != (2,) or velocity.shape != (2,) or not (
                np.isfinite(target).all() and np.isfinite(velocity).all()):
            raise ValueError('Targets must be finite vectors of shape (2,)')
        self.q_target = target.copy()
        raw = (50 * (target - self.data.qpos[self.qadr])
               + 8 * (velocity - self.data.qvel[self.dofs])
               + self.observation.qfrc_bias[self.dofs] - self.observation.qfrc_passive[self.dofs])
        self.last_saturated_motor_channels = int(np.count_nonzero(np.abs(raw) > 10))
        self.saturation_count += self.last_saturated_motor_channels
        self.data.ctrl[self.motors] = np.clip(raw, -10, 10)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            raise SimulationFault('non_finite_state')
        self.last_warnings = [dict(name=mujoco.mjtWarning(i).name,
                                   count=int(self.data.warning[i].number))
                              for i in range(mujoco.mjtWarning.mjNWARNING)
                              if self.data.warning[i].number]
        if self.last_warnings:
            raise SimulationFault('mujoco_warning', dict(warnings=self.last_warnings,
                                                         warning_time_s=float(self.data.time)))
        return self._observe()

    def _observe(self):
        model, data = self.model, self.data
        self.observation = copy.copy(data)
        mujoco.mj_forward(model, self.observation)
        contacts, button_contacts = [], []
        force = np.zeros(6)
        for ci, c in enumerate(self.observation.contact):
            names = [model.geom(int(g)).name for g in (c.geom1, c.geom2)]
            contacts.append(names)
            if self.button in (c.geom1, c.geom2):
                mujoco.mj_contactForce(model, self.observation, ci, force)
                button_contacts.append(dict(geoms=names, distance_m=float(c.dist), normal_force_N=float(force[0])))
        row = dict(step=self.step_count, time_s=float(data.time),
                   slider_position_m=float(data.qpos[self.sq]),
                   slider_velocity_m_s=float(data.qvel[self.sd]),
                   finger_button_distance_m=float(mujoco.mj_geomDistance(
                       model, self.observation, self.finger, self.button, 10, None)),
                   button_contacts=button_contacts, all_contact_pairs=contacts,
                   max_abs_hinge_velocity_rad_s=float(np.max(np.abs(
                       data.qvel[model.jnt_dofadr[model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE)]]))),
                   arm_error_rad=(self.q_target - data.qpos[self.qadr]).tolist(),
                   q_target_rad=self.q_target.tolist(), motor_torque_Nm=data.ctrl[self.motors].tolist())
        loaded = any(set(c['geoms']) == {model.geom(self.finger).name, model.geom(self.button).name}
                     and c['distance_m'] <= 0 and c['normal_force_N'] > 0 for c in button_contacts)
        row['press_task'] = dict(time_s=row['time_s'], position_m=row['slider_position_m'],
                                 contact_loaded=loaded,
                                 **self.task.update(row['time_s'], row['slider_position_m'], loaded))
        return row

    def reference_target(self, step_index):
        """Regression trajectory only; callers can supply their own step targets."""
        t = step_index * self.model.opt.timestep
        phase = np.clip((t - 0.04) / 0.2, 0, 1)
        blend = 3 * phase**2 - 2 * phase**3
        rate = (6 * phase - 6 * phase**2) / 0.2
        if t >= 0.4:
            phase = np.clip((t - 0.4) / 0.2, 0, 1)
            blend = 1 - (3 * phase**2 - 2 * phase**3)
            rate = -(6 * phase - 6 * phase**2) / 0.2
        return self.q_out + blend * (self.q_press - self.q_out), rate * (self.q_press - self.q_out)
