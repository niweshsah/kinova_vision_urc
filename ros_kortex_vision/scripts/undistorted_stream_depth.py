#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# Import the distortion and pinhole camera classes from the provided script
from radialTangentialDistortion import RadialTangentialDistortion, PinholeCamera

class YOLOv8DepthNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('yolov8_depth_node', anonymous=True)

        # Load YOLOv8 model
        self.model = YOLO("yolo-Weights/yolov8n.pt")

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Subscribe to RGB and Depth camera topics
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        # Publisher for annotated image (optional)
        self.image_pub = rospy.Publisher("/yolo/detected_image", Image, queue_size=10)

        # Store latest depth image
        self.depth_image = None
        self.depth_encoding = None

        # Initialize the pinhole camera model with distortion parameters
        self.distortion = RadialTangentialDistortion(-0.3, 0.1, -0.0001, -0.00005)
        self.pinhole_camera = PinholeCamera(640, 480, 450, 450, 319.5, 239.5, self.distortion)

    def depth_callback(self, msg):
        """ Stores the latest depth image """
        try:
            self.depth_encoding = msg.encoding  # Check depth format
            rospy.loginfo(f"Depth image encoding: {self.depth_encoding}")

            # Convert depth image based on encoding
            if self.depth_encoding == "16UC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1") * 0.001  # Convert mm to meters
            elif self.depth_encoding == "32FC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            else:
                rospy.logwarn(f"Unsupported depth encoding: {self.depth_encoding}")
                self.depth_image = None

            rospy.loginfo(f"Depth image shape: {self.depth_image.shape}")

        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")
            self.depth_image = None

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            rospy.loginfo(f"RGB image shape: {cv_image.shape}")

            # Undistort the RGB image using the pinhole camera model
            undistorted_image = self.undistort_image(cv_image)

            # Perform object detection on the undistorted image
            results = self.model(undistorted_image, stream=True)

            # Render detections on the undistorted image
            rendered_image = undistorted_image.copy()  # Start with undistorted image

            if self.depth_image is not None:
                min_depth = np.min(self.depth_image)
                max_depth = np.max(self.depth_image)
                rospy.loginfo(f"Depth Range: {min_depth:.2f}m to {max_depth:.2f}m")

                # Normalize and apply colormap to depth image for visualization
                depth_vis = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = np.uint8(depth_vis)
                depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            height_rgb, width_rgb = undistorted_image.shape[:2]
            height_depth, width_depth = self.depth_image.shape[:2]

            scale_x = width_depth / width_rgb
            scale_y = height_depth / height_rgb

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls != 41:  # Only process "cup"
                        continue

                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Center in RGB image coordinates
                    center_x_rgb, center_y_rgb = (x1 + x2) // 2, (y1 + y2) // 2

                    # Scale to depth image resolution
                    center_x_depth = int(center_x_rgb * scale_x)
                    center_y_depth = int(center_y_rgb * scale_y)

                    # Ensure valid depth image coordinates
                    if 0 <= center_x_depth < width_depth and 0 <= center_y_depth < height_depth:
                        depth_value = self.depth_image[center_y_depth, center_x_depth]
                    else:
                        depth_value = -1  # Invalid depth

                    rospy.loginfo(f"Cup detected at RGB({center_x_rgb}, {center_y_rgb}) -> Depth({center_x_depth}, {center_y_depth}), Depth: {depth_value:.2f}m")

                    # Draw bounding box in RGB image
                    cv2.rectangle(rendered_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(rendered_image, (center_x_rgb, center_y_rgb), 5, (255, 0, 0), -1)  # Blue dot

                    # Draw bounding box in Depth Image
                    cv2.rectangle(depth_vis_colored, (center_x_depth - 10, center_y_depth - 10), 
                                (center_x_depth + 10, center_y_depth + 10), (0, 255, 255), 2)  # Yellow box
                    cv2.circle(depth_vis_colored, (center_x_depth, center_y_depth), 5, (255, 0, 0), -1)  # Blue dot

                    # Depth information text
                    text = f"{depth_value:.2f}m"
                    cv2.putText(rendered_image, text, (x2 - 60, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(depth_vis_colored, text, (center_x_depth, center_y_depth - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Convert and publish annotated image
            annotated_image_msg = self.bridge.cv2_to_imgmsg(rendered_image, "bgr8")
            self.image_pub.publish(annotated_image_msg)

            # Display the annotated images
            cv2.imshow("YOLOv8 Detection - Cups Only", rendered_image)
            cv2.imshow("Depth Image with Detections", depth_vis_colored)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def undistort_image(self, image):
        """ Undistort the image using the pinhole camera model """
        h, w = image.shape[:2]
        map1, map2 = cv2.initUndistortRectifyMap(
            self.pinhole_camera.camera_matrix, self.pinhole_camera.dist_coeffs, None, None, (w, h), cv2.CV_32FC1
        )
        undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        return undistorted_image

if __name__ == '__main__':
    try:
        yolo_node = YOLOv8DepthNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass