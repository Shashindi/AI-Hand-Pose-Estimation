import os
import uuid
import sys
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def main(save_frames: bool = False, save_dir: str = "Output Images") -> None:
	if save_frames:
		os.makedirs(save_dir, exist_ok=True)

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open webcam. Ensure camera permissions are granted and no other app is using it.")

	with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break

			# BGR to RGB and horizontal flip
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image = cv2.flip(image, 1)

			# Inference
			image.flags.writeable = False
			results = hands.process(image)
			image.flags.writeable = True

			# RGB back to BGR
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			# Draw landmarks
			if results.multi_hand_landmarks:
				for hand in results.multi_hand_landmarks:
					mp_drawing.draw_landmarks(
						image,
						hand,
						mp_hands.HAND_CONNECTIONS,
						mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
						mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
					)

			# Optionally save the frame continuously when --save is provided
			if save_frames:
				cv2.imwrite(os.path.join(save_dir, f"{uuid.uuid1()}.jpg"), image)

			cv2.imshow("Hand Tracking", image)

			key = cv2.waitKey(10) & 0xFF
			if key == ord("q"):
				break
			elif key == ord("s"):
				# Save a single snapshot on 's' key press
				os.makedirs(save_dir, exist_ok=True)
				cv2.imwrite(os.path.join(save_dir, f"{uuid.uuid1()}.jpg"), image)

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	save = "--save" in sys.argv
	main(save_frames=save) 