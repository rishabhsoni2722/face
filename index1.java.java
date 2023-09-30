import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

public class FaceDetection {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the Haar cascade classifier for face detection
        CascadeClassifier faceCascade = new CascadeClassifier("C:\\cv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");

        // Load your face recognition model (Siamese network or ArcFace)
        // You will need to use the appropriate library for this.

        // Read an image
        Mat image = Imgcodecs.imread("C:\\Users\\91961\\OneDrive\\Pictures\\morphing img1.jpg");

        // Convert the image to grayscale for face detection
        Mat grayImage = new Mat();
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);

        // Perform face detection
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(
                grayImage,
                faces,
                1.1,
                3,
                Objdetect.CASCADE_SCALE_IMAGE,
                new Size(30, 30),
                new Size(500, 500)
        );

        // Loop through detected faces and perform recognition
        for (Rect rect : faces.toArray()) {
            // Extract the face region
            Mat faceROI = new Mat(grayImage, rect);

            // Perform face recognition using your pre-trained model

            // Display or save the results
            Imgproc.rectangle(image, rect,new Scalar(0, 255, 0), 2);
        }

        Imgcodecs.imwrite("output.jpg", image);
    }
}
