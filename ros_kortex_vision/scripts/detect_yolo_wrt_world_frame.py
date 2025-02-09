#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from geometry_msgs.msg import PointStamped

class YOLOv8ROSNode:
    def __init__(self):
        rospy.loginfo("Waiting for TF Transform...")
        rospy.sleep(2)  # Wait for TF buffer to populate
        rospy.init_node('yolov8_ros_node', anonymous=True)
        self.model = YOLO('yolov8n.pt')  # Load YOLOv8 Nano model
        self.bridge = CvBridge()

        # Camera intrinsic parameters
        self.color_info = {
            'K': [1297.672904, 0.0, 620.914026, 0.0, 1298.631344, 238.280325, 0.0, 0.0, 1.0]
        }
        
        self.depth_info = {
            'K': [360.01333, 0.0, 243.87228, 0.0, 360.013366699, 137.9218444, 0.0, 0.0, 1.0]
        }

        # ROS Subscribers and Publishers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.image_pub = rospy.Publisher("/yolo/detected_image", Image, queue_size=10)

        self.depth_image = None

        # Initialize TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def depth_callback(self, msg):
        """Process depth image."""
        try:
            if msg.encoding == "16UC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) * 0.001
            elif msg.encoding == "32FC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            rospy.logerr(f"Depth processing error: {e}")

    def compute_3d_coordinates(self, u, v, depth):
        """Compute 3D coordinates from depth."""
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            rospy.logwarn(f"Invalid depth value at ({u}, {v}): {depth}")
            return None

        K = np.array(self.depth_info['K']).reshape(3, 3)
        uv1 = np.array([u, v, 1.0])
        xyz = np.linalg.inv(K) @ uv1 * depth
        return xyz[0], xyz[1], xyz[2]

    
    def get_camera_position(self):
        """Get the current position of the depth camera frame in the world (base_link) frame."""
        try:
            transform = self.tf_buffer.lookup_transform("base_link", "camera_depth_frame", rospy.Time(0), rospy.Duration(1.0))
            
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            return x, y, z
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF Transform Error (Camera Position): {e}")
            return None


    def transform_point(self, camera_point, camera_frame="camera_depth_frame"):
        """Transform a point from the camera frame to the world frame (base_link)."""
        try:
            transform = self.tf_buffer.lookup_transform("base_link", camera_frame, rospy.Time(0), rospy.Duration(1.0))
            
            # rospy.loginfo(f"Transform: {transform}")

            point_stamped = PointStamped()
            point_stamped.header.frame_id = camera_frame
            point_stamped.header.stamp = rospy.Time.now()
            point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = camera_point

            world_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return world_point.point.x, world_point.point.y, world_point.point.z
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            # rospy.logerr(f"TF Transform Error: {e}")
            return None


    def image_callback(self, msg):
            """Process the color image and apply YOLOv8."""
            camera_coords = None
            world_coords = None  # Initialize world_coords before use
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                results = self.model(cv_image)  # YOLOv8 inference
                rendered_image = cv_image.copy()

                if self.depth_image is not None:
                    depth_vis = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
                    depth_vis = cv2.applyColorMap(np.uint8(depth_vis), cv2.COLORMAP_JET)

                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                            center_rgb = ((x1 + x2) // 2, (y1 + y2) // 2)

                            # Project RGB center to depth frame
                            u_rgb, v_rgb = center_rgb
                            u_depth = int(((u_rgb - self.color_info['K'][2]) * 
                                        (self.depth_info['K'][0] / self.color_info['K'][0])) + 
                                        self.depth_info['K'][2])
                            v_depth = int(((v_rgb - self.color_info['K'][5]) * 
                                        (self.depth_info['K'][4] / self.color_info['K'][4])) + 
                                        self.depth_info['K'][5])

                            if 0 <= u_depth < self.depth_image.shape[1] and 0 <= v_depth < self.depth_image.shape[0]:
                                depth_value = self.depth_image[v_depth, u_depth]

                                if not np.isnan(depth_value) and not np.isinf(depth_value):
                                    camera_coords = self.compute_3d_coordinates(u_depth, v_depth, depth_value)
                                    if camera_coords:
                                        world_coords = self.transform_point(camera_coords)
                                        if world_coords:
                                            x, y, z = world_coords
                                        else:
                                            x, y, z = None, None, None
                                    else:
                                        x, y, z = None, None, None
                                else:
                                    x, y, z = None, None, None
                            else:
                                x, y, z = None, None, None

                            # Get the class label and index
                            class_id = int(box.cls[0])
                            class_label = self.model.names[class_id]

                            # Pass the class label and index to draw on the image
                            self.draw_detections(rendered_image, depth_vis, (x1, y1, x2, y2), center_rgb, camera_coords, world_coords, class_label, class_id)

                
                self.publish_and_display(rendered_image, depth_vis)
            
            except Exception as e:
                rospy.logerr(f"Image processing error: {e}")
                
            

    def draw_detections(self, rgb_img, depth_img, bbox, center_rgb, camera_coords, world_coords, class_label, class_id):
        """Draw bounding boxes, display object coordinates, and show camera position."""
        cv2.rectangle(rgb_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.circle(rgb_img, center_rgb, 5, (255, 0, 0), -1)
        
        # Display class label and index
        label_text = f"{class_label} ({class_id})"
        cv2.putText(rgb_img, label_text, (bbox[0], bbox[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if camera_coords is None:
            rospy.logwarn("Camera Coords is None")

        if camera_coords is not None:
            x_cam, y_cam, z_cam = camera_coords
            cv2.putText(rgb_img, f"Cam: ({x_cam:.2f}, {y_cam:.2f}, {z_cam:.2f})m", 
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if world_coords is not None:
            x_world, y_world, z_world = world_coords
            cv2.putText(rgb_img, f"World: ({x_world:.2f}, {y_world:.2f}, {z_world:.2f})m", 
                        (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(rgb_img, "World Coord Unavailable", (bbox[0], bbox[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Get camera position in world coordinates
        camera_position = self.get_camera_position()
        if camera_position is not None:
            x_cam_frame, y_cam_frame, z_cam_frame = camera_position
            cv2.putText(rgb_img, f"Cam Pos: ({x_cam_frame:.2f}, {y_cam_frame:.2f}, {z_cam_frame:.2f})m", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)



    def publish_and_display(self, rgb_img, depth_img):
        """Publish processed images and show them in OpenCV windows."""
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(rgb_img, "bgr8"))
        cv2.imshow("YOLOv8 Detection", rgb_img)
        cv2.imshow("Depth View", depth_img)
        cv2.waitKey(1)

if __name__ == '__main__':
    YOLOv8ROSNode()
    rospy.spin()
