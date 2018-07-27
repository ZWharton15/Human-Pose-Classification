"""
Wrap the OpenPose library with Python.
To install run `make install` and library will be stored in /usr/local/python
"""
import numpy as np
import ctypes as ct
import cv2
import os
import math
from sys import platform
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
dir_path = os.path.dirname(os.path.realpath(__file__))

if platform == "win32":
    os.environ['PATH'] = dir_path + "/../../bin;" + os.environ['PATH']
    os.environ['PATH'] = dir_path + "/../../x64/Debug;" + os.environ['PATH']
    os.environ['PATH'] = dir_path + "/../../x64/Release;" + os.environ['PATH']

class OpenPose(object):
    """
    Ctypes linkage
    """
    if platform == "linux" or platform == "linux2":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.so')
    elif platform == "darwin":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.dylib')
    elif platform == "win32":
        try:
            _libop= np.ctypeslib.load_library('_openpose', dir_path+'/Release/_openpose.dll')
        except OSError as e:
            _libop= np.ctypeslib.load_library('_openpose', dir_path+'/Debug/_openpose.dll')
    _libop.newOP.argtypes = [
        ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_float, ct.c_float, ct.c_int, ct.c_float, ct.c_int, ct.c_bool, ct.c_char_p]
    _libop.newOP.restype = ct.c_void_p
    _libop.delOP.argtypes = [ct.c_void_p]
    _libop.delOP.restype = None

    _libop.forward.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.uint8), ct.c_bool]
    _libop.forward.restype = None

    _libop.getOutputs.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.getOutputs.restype = None

    _libop.poseFromHeatmap.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.uint8),
        np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.poseFromHeatmap.restype = None

    def encode(self, string):
        return ct.c_char_p(string.encode('utf-8'))

    def __init__(self, params):
        """
        OpenPose Constructor: Prepares OpenPose object

        Parameters
        ----------
        params : dict of required parameters. refer to openpose example for more details

        Returns
        -------
        outs: OpenPose object
        """
        self.op = self._libop.newOP(params["logging_level"],
		                            self.encode(params["output_resolution"]),
                                    self.encode(params["net_resolution"]),
                                    self.encode(params["model_pose"]),
                                    params["alpha_pose"],
                                    params["scale_gap"],
                                    params["scale_number"],
                                    params["render_threshold"],
                                    params["num_gpu_start"],
                                    params["disable_blending"],
                                    self.encode(params["default_model_folder"]))

    def __del__(self):
        """
        OpenPose Destructor: Destroys OpenPose object
        """
        self._libop.delOP(self.op)

    def forward(self, image, display = False):
        """
        Forward: Takes in an image and returns the human 2D poses, along with drawn image if required

        Parameters
        ----------
        image : color image of type ndarray
        display : If set to true, we return both the pose and an annotated image for visualization

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(3),dtype=np.int32)
        self._libop.forward(self.op, image, shape[0], shape[1], size, displayImage, display)
        array = np.zeros(shape=(size),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        if display:
            return array, displayImage
        return array

    def poseFromHM(self, image, hm, ratios=[1]):
        """
        Pose From Heatmap: Takes in an image, computed heatmaps, and require scales and computes pose

        Parameters
        ----------
        image : color image of type ndarray
        hm : heatmap of type ndarray with heatmaps and part affinity fields
        ratios : scaling ration if needed to fuse multiple scales

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
        if len(ratios) != len(hm):
            raise Exception("Ratio shape mismatch")

        # Find largest
        hm_combine = np.zeros(shape=(len(hm), hm[0].shape[1], hm[0].shape[2], hm[0].shape[3]),dtype=np.float32)
        i=0
        for h in hm:
           hm_combine[i,:,0:h.shape[2],0:h.shape[3]] = h
           i+=1
        hm = hm_combine

        ratios = np.array(ratios,dtype=np.float32)

        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(4),dtype=np.int32)
        size[0] = hm.shape[0]
        size[1] = hm.shape[1]
        size[2] = hm.shape[2]
        size[3] = hm.shape[3]

        self._libop.poseFromHeatmap(self.op, image, shape[0], shape[1], displayImage, hm, size, ratios)
        array = np.zeros(shape=(size[0],size[1],size[2]),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        return array, displayImage

    @staticmethod
    def process_frames(frame, boxsize = 368, scales = [1]):
        base_net_res = None
        imagesForNet = []
        imagesOrig = []
        for idx, scale in enumerate(scales):
            # Calculate net resolution (width, height)
            if idx == 0:
                net_res = (16 * int((boxsize * frame.shape[1] / float(frame.shape[0]) / 16) + 0.5), boxsize)
                base_net_res = net_res
            else:
                net_res = ((min(base_net_res[0], max(1, int((base_net_res[0] * scale)+0.5)/16*16))),
                          (min(base_net_res[1], max(1, int((base_net_res[1] * scale)+0.5)/16*16))))
            input_res = [frame.shape[1], frame.shape[0]]
            scale_factor = min((net_res[0] - 1) / float(input_res[0] - 1), (net_res[1] - 1) / float(input_res[1] - 1))
            warp_matrix = np.array([[scale_factor,0,0],
                                    [0,scale_factor,0]])
            if scale_factor != 1:
                imageForNet = cv2.warpAffine(frame, warp_matrix, net_res, flags=(cv2.INTER_AREA if scale_factor < 1. else cv2.INTER_CUBIC), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            else:
                imageForNet = frame.copy()

            imageOrig = imageForNet.copy()
            imageForNet = imageForNet.astype(float)
            imageForNet = imageForNet/256. - 0.5
            imageForNet = np.transpose(imageForNet, (2,0,1))

            imagesForNet.append(imageForNet)
            imagesOrig.append(imageOrig)

        return imagesForNet, imagesOrig

    @staticmethod
    def draw_all(imageForNet, heatmaps, currIndex, div=4., norm=False):
        netDecreaseFactor = float(imageForNet.shape[0]) / float(heatmaps.shape[2]) # 8
        resized_heatmaps = np.zeros(shape=(heatmaps.shape[0], heatmaps.shape[1], imageForNet.shape[0], imageForNet.shape[1]))
        num_maps = heatmaps.shape[1]
        combined = None
        for i in range(0, num_maps):
            heatmap = heatmaps[0,i,:,:]
            resizedHeatmap = cv2.resize(heatmap, (0,0), fx=netDecreaseFactor, fy=netDecreaseFactor)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resizedHeatmap)

            if i==currIndex and currIndex >=0:
                resizedHeatmap = np.abs(resizedHeatmap)
                resizedHeatmap = (resizedHeatmap*255.).astype(dtype='uint8')
                im_color = cv2.applyColorMap(resizedHeatmap, cv2.COLORMAP_JET)
                resizedHeatmap = cv2.addWeighted(imageForNet, 1, im_color, 0.3, 0)
                cv2.circle(resizedHeatmap, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
                return resizedHeatmap
            else:
                resizedHeatmap = np.abs(resizedHeatmap)
                if combined is None:
                    combined = np.copy(resizedHeatmap);
                else:
                    if i <= num_maps-2:
                        combined += resizedHeatmap;
                        if norm:
                            combined = np.maximum(0, np.minimum(1, combined));

        if currIndex < 0:
            combined /= div
            combined = (combined*255.).astype(dtype='uint8')
            im_color = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
            combined = cv2.addWeighted(imageForNet, 0.5, im_color, 0.5, 0)
            cv2.circle(combined, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
            return combined

'''
Utilises the OpenPose Python API to extract the skeleton data of an image and classify the arm positioning of each person in view.
The model was trained using the VGG16 architecture on 36 different arm poses recorded with the camera at eye level of the participant.
'''

#Method to draw each of the bones of the skeletons and return that image
def visialize_person(img, person, pairs):
    stickwidth = 4
    cur_img = img.copy()
    counter = 0;
    for i in range(0, len(pairs),2):   
        if person[pairs[i],0] > 0 and person[pairs[i+1],0] > 0:
            Y = [person[pairs[i],0], person[pairs[i+1],0]]
            X = [person[pairs[i],1], person[pairs[i+1],1]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_img, polygon, colours[counter])
            counter = counter + 1
    
    img = cv2.addWeighted(img, 0.4, cur_img, 0.6, 0)
    return img

#Method to crop the final image to the skeleton area
def crop_person(img4, person, name):
    #Values ot store the range of the skeleton in each axis
    minx = 99999
    miny = 99999
    maxx = 0
    maxy = 0
    counter = 0
    partial_joints = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
    
    #Go through each joint to find the min and max coords, ignoring 0 or irrelevant joint coordinates
    for joint in person:
        x, y, score = joint
        if counter in partial_joints:
            if x > 0:
                if round(x) > maxx:
                    maxx = x + 10
                if round(x) < minx:
                    minx = x - 10
                if round(y) > maxy:
                    maxy = y + 10
                if round(y) < miny:
                    miny = y - 10
        counter += 1
    #Crop the image using OpenCV
    cropped = img4[int(round(miny)):int(round(maxy)), int(round(minx)):int(round(maxx))]
    #Save the image
    cv2.imwrite(os.path.join(name, "skeleton.jpg"), cropped)
    pass

#Method to draw the skeleton used in the prediction stage
def draw_person(base_img, index, keypoints, output_dir):
    #Get current person data
    person = keypoints[index]
    #Joint elements required for the skeleton in the person list
    partial_joints = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]
    #Make a blank image used to draw the skeleton overlay
    part_skele_img = np.zeros((base_img.shape[0],base_img.shape[1],3), np.uint8)
    counter = 0
    
    #Iterate through each joint and draw it on screen if it is in the above list
    for z in person:
        x, y, score = z
        if x > 0  and y > 0:
            if counter in partial_joints:
                cv2.circle(part_skele_img, (round(x), round(y)), 7, (0, 0, 255), -1)
        counter += 1
    #Draw the bones of the skeleton
    part_skele_img = visialize_person(part_skele_img, person, part_pairs)
    
    #Crop the image so the input data is smaller and more precise
    crop_person(part_skele_img, person, output_dir)
    pass

#Method to predict the pose from a skeleton image
def predict_pose(model, img_path):
    #The default image size of the InceptionV3 model
    dim = 299
    #Load the newly created skeleton image
    img = image.load_img(img_path, target_size=(dim, dim))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #Predict the class
    pred = model.predict(x)
    pred = pred.tolist()
    pred = pred[0]
    #Return the prediction with the highest confidence
    return pred.index(max(pred))

#Convert the class index to a name
def label_to_name(legend, label_num):
    #read the conversion file to get the list index
    with open(legend, 'r') as f:
        lines = f.readlines()
        return lines[int(label_num)].split(",")[1]

#Method to process the image using OpenPose and the custom Keras model
def get_pose(frame, model, base_dirname):
    #Get the skeleton data from OpenPose
    keypoints = openpose.forward(frame)
    #Count how many people were found
    num_people = len(keypoints)
    print(f"{num_people} Skeletons Found...")
    
    if num_people == 0:
        print("No People Found\n\n\n.")
        return
    else:
        #Iterate through each person to do multi pose classification
        for current_person in range(0, num_people):
            #Draw the skeleton data for use in the model
            draw_person(frame, current_person, keypoints, os.path.join(base_dirname, 'data'))
            print("Skeleton Extracted...")
            #Get the class number of the pose (0-35)
            label = predict_pose(model, os.path.join(base_dirname, 'data', "skeleton.jpg"))
            print("Pose predicted!")
            #Convert the value to an actual class name and write it
            print(label_to_name(os.path.join(base_dirname, 'legend.txt'), label))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label_to_name(os.path.join(base_dirname, 'legend.txt'), label), (30,60), font, 1, (255,0,255), 2, cv2.LINE_AA)
            
    #Show the basic image with no alterations made to it
    cv2.imshow("Pose Classification", frame)

#Main method which takes parameters to determine if openpose should be given a webcam or a directory of images
def pose_classification(use_webcam=True, image_dir=None):
    #Gets the directory of this script for use to find the model and temporary output image of each iteration
    base_dirname = os.path.dirname(__file__)
    #Load the model in Keras
    print("Loading Model...")
    model = load_model(os.path.join(base_dirname, 'model', 'i3_pose_model.hdf5'))
    print("...Model Loaded")
    if use_webcam:
        #Get the webcam output in an infinte loop
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            #Call method to predict the pose
            try:
                get_pose(frame, model, base_dirname)
            except:
                print("Frame skipped.")
            #Only break the loop and end the program when the 'q' key is pressed
            if cv2.waitKey(1) & ord('q'):
                break
    else:
        #Read all images from the directory parameter
        for image_name in os.listdir(image_dir):
            print(f"Processing: {image_name}")
            img = cv2.imread(os.path.join(image_dir, image_name))
            #Call method to predict the pose
            print("Estimating Pose...")
            try:
            get_pose(img, model, base_dirname)
            except:
                print("Frame skipped.")
            #Wait for the 'Enter' key to be pressed after each image
            print("Press enter to continue.")
            cv2.waitKey(1)

#Colours and joint values for OpenPose used when drawing the skeleton
colours = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
part_pairs = [1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,0,   0,15,   15,17]


if __name__ == "__main__":
    #Default openpose parameters found on the Python API script
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    params["default_model_folder"] = "../../../models/"
    openpose = OpenPose(params)
    
    #The main method call for this project
    #use_webcam - Boolean value that determines if openpose will run from the webcam or not (if this is False then image_dir must be a filepath to the image directory)
    #image_dir - String value that points to the parent directory of all images to be processed (not required if use_webcam is True)
    pose_classification(use_webcam=False, image_dir="DIRECTORY OF IMAGES")