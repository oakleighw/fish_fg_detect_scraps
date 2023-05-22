%Background mixture models (Background Subtraction)
%Stauffer C, Grimson W.E.L (1999)
%http://www.ai.mit.edu/projects/vsam/Publications/stauffer_cvpr98_track.pdf
%"Pixel values that donot match one of the pixels “background” Gaussiansare grouped using connected components. Finally,the connected components are tracked from frame to frame using a multiple hypothesis tracker.

%KaewTraKulPong P, Bowden R (2001)
%http://www.ee.surrey.ac.uk/CVSSP/Publications/papers/KaewTraKulPong-AVBS01.pdf
%Improves apon 1999 method, more adaptive to environments and does not
%include object shadows. Speedier learning & accuracy. Adaptive Gaussian
%Mixture Model.

%MATLAB https://uk.mathworks.com/help/vision/ref/vision.foregrounddetector-system-object.html

clear,clc,close all;

videoSource = VideoReader('C:\Users\Computing\OneDrive - University of Lincoln (1)\Documents\1fish_tracking\dataset\pooled_Vids\30_05seatrout.mp4');


%C:\Users\Computing\OneDrive - University of Lincoln
%(1)\Documents\1fish_tracking\dataset\wetransfer-b9b88f\wetransfer-b9b88f\multi_trackdata_61ac.mp4

detector = vision.ForegroundDetector(...
       'NumTrainingFrames', 15, ...
       'InitialVariance', 30*30);

blob = vision.BlobAnalysis(...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 250);


%add border around fish
shapeInserter = vision.ShapeInserter('BorderColor','White');

videoPlayer = vision.VideoPlayer();

% vidName = "outputs/1_6_salmon_track";
% fname = vidName + "_bg_subtraction";



vidName = "outputs/1_6_salmon_track";
fname = vidName + "_multi_MOG_bw";
v = VideoWriter(fname);
open(v)

while hasFrame(videoSource)
     frame  = readFrame(videoSource);
     %grey = im2gray(frame); %
     %contrasted = imadjust(grey);%
     %mask = imbinarize(grey, 'adaptive');
     %regionfills to reduce impact of noise
     %regionfill smoothly interpolates inward from the pixel values on the
     %outer boundary of the regions. regionfill calculates the discrete
     %Laplacian over the regions and solves the Dirichlet boundary value
     %problem - MATLAB.
     %repairedImage = regionfill(grey, mask);


     % fgMask = detector(repairedImage);
     % bbox   = blob(fgMask);
     % out    = shapeInserter(repairedImage,bbox);

     fgMask = detector(frame);
     bbox   = blob(fgMask);
     out    = fgMask;

     
     writeVideo(v,double(out));
     videoPlayer(out);
     pause(0.1);
end

close(v)
release(videoPlayer);