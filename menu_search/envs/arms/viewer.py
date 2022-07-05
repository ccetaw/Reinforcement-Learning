from aitviewer.renderables.lines import Lines
from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer
import numpy as np


class ArmViewer(Viewer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene.camera.position = np.array([0.0, 1.3, 5.0])
        self.run_animations = True

    def setup_run(self, shoulder, elbow, wrist):
        shoulder = self._convert_coordinate(shoulder)
        wrist = self._convert_coordinate(wrist)
        elbow = self._convert_coordinate(elbow)

        line_strip = np.zeros((120, 4, 3))
        line_strip[:, 0, :] = shoulder[:, 0, :]
        line_strip[:, 1, :] = elbow[:, 0, :]
        line_strip[:, 2, :] = elbow[:, 0, :]
        line_strip[:, 3, :] = wrist[:, 0, :]

        lines = Lines(line_strip, mode='lines', r_base=0.01)
        s_sphere = Spheres(shoulder, color=(1.0, 0.0, 1.0, 1.0), radius=0.2)
        e_sphere = Spheres(elbow, color=(1.0, 0.0, 1.0, 1.0), radius=0.1)
        w_sphere = Spheres(wrist, color=(1.0, 0.0, 1.0, 1.0), radius=0.05)
        self.scene.add(s_sphere, w_sphere, e_sphere, lines)

    def _convert_coordinate(self, coordinates):
        transform_coordinates = np.zeros(coordinates.shape)
        transform_coordinates[:, :, 0] = coordinates[:, :, 0]
        transform_coordinates[:, :, 1] = coordinates[:, :, 2]
        transform_coordinates[:, :, 2] = coordinates[:, :, 1]
        return transform_coordinates


if __name__ == '__main__':
    # Display in viewer.

    n_frames = 120
    shoulder = np.zeros((n_frames, 1, 3))
    wrist = np.zeros((n_frames, 1, 3))
    elbow = np.zeros((n_frames, 1, 3))
    for i in range(n_frames):
        elbow[i, 0, :] = [0, 0.05 * i, 0]
        wrist[i, 0, :] = [0, 0, 0.05 * i]
    v = ArmViewer()
    v.setup_run(shoulder, elbow, wrist)
    v.run()
