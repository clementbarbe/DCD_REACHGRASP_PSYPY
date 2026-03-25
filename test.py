import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened()
    ret, frame = cap.read() if ok else (False, None)
    shape = None if frame is None else frame.shape
    print(f"index={i} opened={ok} read={ret} shape={shape}")
    cap.release()
