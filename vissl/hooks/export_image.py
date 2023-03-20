from classy_vision.hooks.classy_hook import ClassyHook
import os
import torchvision

# @register_hook("ExportImageHook")
class ExportImageHook(ClassyHook):
    on_start = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir


    def on_update(self, state):
        self.export_images(state)

    def export_images(self, task):
        input_data = task.last_batch.sample["input"]
        current_iter = task.iteration
        for idx, image in enumerate(input_data):
            # Normalize the image to the [0, 1] range if necessary
            image = (image - image.min()) / (image.max() - image.min())
            file_path = os.path.join(self.output_dir, f"{current_iter}_{idx}.png")
            torchvision.utils.save_image(image, file_path)
