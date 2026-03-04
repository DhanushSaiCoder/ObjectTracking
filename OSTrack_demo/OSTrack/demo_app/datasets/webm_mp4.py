import cv2

def convert_webm_to_mp4_cv2(input_file, output_file, fps=30):
    cap = cv2.VideoCapture(input_file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps > 0:
        fps = original_fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # You can modify frame here
        out.write(frame)

    cap.release()
    out.release()
    print("Done")

convert_webm_to_mp4_cv2("cars.webm", "../assets/cars.mp4")