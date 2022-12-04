import imageio
import os


class VideoRecorder(object):
    def __init__(self, dir_name, height=448, width=448, camera_id=0, fps=25):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, camera="static", mode=None, domain_name="robot"):
        if self.enabled:
            if domain_name == "robot":
                frame = env.unwrapped.render_obs(
                    mode='rgb_array',
                    height=self.height,
                    width=self.width,
                    camera_id="camera_dynamic"
                )
            elif domain_name == "metaworld":
                frame = env.render_obs(
                    mode='rgb_array',
                    height=self.height,
                    width=self.width,
                    camera_id="camera_dynamic"
                )


            if camera is None:
                from torchvision.utils import make_grid
                import torch
                frame = make_grid( torch.from_numpy(frame).permute(0, 3, 1, 2), n_rows=2)
                frame = frame.permute(1, 2, 0)
            elif camera == "static":
                frame = frame[0] 
            elif camera == "dynamic":
                frame = frame[1]
            else:
                raise Exception("error camera mode")

            if mode is not None and 'video' in mode:
                _env = env
                while 'video' not in _env.__class__.__name__.lower():
                    _env = _env.env
                frame = _env.apply_to(frame)
            self.frames.append(frame)
            
    def record_frame(self, frame):
        self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
