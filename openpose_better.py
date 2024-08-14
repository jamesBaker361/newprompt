from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3,resize_image
import numpy as np

class OpenPoseDetectorProbs(OpenposeDetector):
    def probs(self, input_image, detect_resolution=512, include_hand=False, include_face=False)->float:
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        
        poses = self.detect_poses(input_image, include_hand, include_face)

        if len(poses)==0:
            return 0.0
        
        total_score=poses[0].body.total_score
        total_parts=poses[0].body.total_parts

        

        return float(total_score)/total_parts
    