from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3,resize_image
from controlnet_aux.open_pose.util import draw_bodypose
from controlnet_aux.open_pose.body import Keypoint
from controlnet_aux.open_pose import PoseResult,BodyResult
from typing import List,Union
import numpy as np
from PIL import Image,ImageDraw
import cv2

def normalize_keypoints(keypoints: List[Keypoint]) -> List[Keypoint]:
    xs = [kp.x for kp in keypoints]
    ys = [kp.y for kp in keypoints]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height = max_x - min_x, max_y - min_y

    normalized_keypoints = [
        Keypoint((kp.x - min_x) / width, (kp.y - min_y) / height, kp.score, kp.id)
        for kp in keypoints
    ]
    
    return normalized_keypoints

def calculate_transformation(source: List[Keypoint], target: List[Keypoint]):
    source_xs = np.array([kp.x for kp in source])
    source_ys = np.array([kp.y for kp in source])
    target_xs = np.array([kp.x for kp in target])
    target_ys = np.array([kp.y for kp in target])
    
    scale_x = (target_xs.max() - target_xs.min()) / (source_xs.max() - source_xs.min())
    scale_y = (target_ys.max() - target_ys.min()) / (source_ys.max() - source_ys.min())
    
    scale = (scale_x + scale_y) / 2
    
    trans_x = target_xs.mean() - source_xs.mean() * scale
    trans_y = target_ys.mean() - source_ys.mean() * scale
    
    return scale, trans_x, trans_y

def transform_keypoints(keypoints: List[Keypoint], scale: float, trans_x: float, trans_y: float) -> List[Keypoint]:
    transformed_keypoints = [
        Keypoint(kp.x * scale + trans_x, kp.y * scale + trans_y, kp.score, kp.id)
        for kp in keypoints
    ]
    
    return transformed_keypoints

def adjust_keypoints(shape_keypoints: List[Keypoint], proportion_keypoints: List[Keypoint]) -> List[Keypoint]:
    normalized_shape = normalize_keypoints(shape_keypoints)
    normalized_proportion = normalize_keypoints(proportion_keypoints)
    
    scale, trans_x, trans_y = calculate_transformation(normalized_proportion, normalized_shape)
    
    adjusted_keypoints = transform_keypoints(proportion_keypoints, scale, trans_x, trans_y)
    
    return adjusted_keypoints

class OpenposeDetectorResize(OpenposeDetector):
    def __call__(
            self,
            proportion_image:np.ndarray,
            shape_image:np.ndarray,
            detect_resolution:int=512,
              image_resolution:int=512, 
              include_body:bool=True, 
              include_hand:bool=False, 
              include_face:bool=False,
               output_type="pil"):
        if not isinstance(proportion_image, np.ndarray):
            proportion_image = np.array(proportion_image, dtype=np.uint8)
        if not isinstance(shape_image, np.ndarray):
            shape_image = np.array(shape_image, dtype=np.uint8)

        proportion_image = HWC3(proportion_image)
        proportion_image = resize_image(proportion_image, detect_resolution)
        H, W, C = proportion_image.shape

        proportion_poses = self.detect_poses(proportion_image, include_hand, include_face)

        proportion_keypoints=proportion_poses[0].body.keypoints

        shape_image = HWC3(shape_image)
        shape_image = resize_image(shape_image, detect_resolution)
        H, W, C = shape_image.shape

        shape_poses = self.detect_poses(shape_image, include_hand, include_face)

        shape_keypoints=shape_poses[0].body.keypoints

        adjusted=adjust_keypoints(shape_keypoints, proportion_keypoints)

        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        canvas = draw_bodypose(canvas, adjusted)

        detected_map = canvas
        detected_map = HWC3(detected_map)

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
    

def get_poseresult(self:OpenposeDetector,
        image:np.ndarray,
        detect_resolution=512,
        include_hand=False,
        include_face=True
        )->PoseResult:
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.uint8)

    image = HWC3(image)
    image = resize_image(image, detect_resolution)

    return self.detect_poses(image, include_hand, include_face)[0]
    
def intermediate_points_body(keypoints:List[Union[Keypoint, None]],n_points=1)->List[Keypoint]:
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]
    new_points=[]
    for (k1_index, k2_index) in limbSeq:
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        total_x_dist=keypoint2.x-keypoint1.x
        total_y_dist=keypoint2.y-keypoint1.y

        step_size_x=total_x_dist/(n_points+1)
        step_size_y=total_y_dist/(n_points+1)

        for i in range(1,1+n_points):
            new_points.append(Keypoint(keypoint1.x+i*step_size_x,keypoint1.y+i*step_size_y))
    
    return new_points

if __name__=='__main__':
    for n in [1,2,3]:
        src=Image.open("percy.jpg")
        draw = ImageDraw.Draw(src)
        print(src.size)
        H,W=src.size
        detector=OpenposeDetectorResize.from_pretrained('lllyasviel/ControlNet')
        points=detector.get_points(src)
        print(len(points))
        for i,k in enumerate(points[0].body.keypoints):
            if k is not None:
                
                x=k.x*H
                y=k.y*W
                radius = 4

                print(x,y)

                # Draw a small red circle
                draw.ellipse(
                    (x - radius, y - radius, 
                    x+ radius, y+ radius), 
                    fill='red', outline='red'
                )
            else:
                print(f"point at index {i} is none")

        

        for kp in intermediate_points_body(points[0].body.keypoints,n_points=n):
            break
            x=kp.x*H
            y=kp.y*W
            radius = 4

            print(x,y)

            # Draw a small red circle
            draw.ellipse(
                (x - radius, y - radius, 
                x+ radius, y+ radius), 
                fill='purple', outline='purple'
            )

        for i,k in enumerate(points[0].face):
            if k is not None:
                
                x=k.x*H
                y=k.y*W
                radius = 1

                print(x,y)

                # Draw a small red circle
                draw.ellipse(
                    (x - radius, y - radius, 
                    x+ radius, y+ radius), 
                    fill='green', outline='green'
                )
            else:
                print(f"face point at index {i} is none")

        src.save(f"new_percy_{n}.jpg")