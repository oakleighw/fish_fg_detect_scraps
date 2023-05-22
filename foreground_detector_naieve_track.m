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

videoSource = VideoReader('C:\Users\Computing\OneDrive - University of Lincoln (1)\Documents\1fish_tracking\dataset\wetransfer-ac9d73\wetransfer-ac9d73\1_6_salmon.mp4');

detector = vision.ForegroundDetector(...
       'NumTrainingFrames', 15, ...
       'InitialVariance', 30*30);

blob = vision.BlobAnalysis(...
       'CentroidOutputPort', false, 'AreaOutputPort', false, ...
       'BoundingBoxOutputPort', true, ...
       'MinimumBlobAreaSource', 'Property', 'MinimumBlobArea', 250);


%add border around fish
shapeInserter = vision.ShapeInserter('BorderColor','White');


%shapeInserter2 = vision.ShapeInserter('line',[0 1280], [500 500],Color='white');


videoPlayer = vision.VideoPlayer();

fishcount = 0;

vidName = "outputs/1_6_salmon_track";
fname = vidName + "_bg_subtraction";
v = VideoWriter(fname);
open(v);



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
     fgMask = detector(frame);
     bbox   = blob(fgMask);
     disp(bbox);

     if bbox
         if (bbox(2)) > 410
             disp("Counted");
             fishcount = fishcount + 1;
         end
     end

     out    = shapeInserter(frame,bbox);
     out    = insertShape(out,'Line',[[1300 450], [0 500]]);
     out    = insertText(out,[10,10], fishcount);
     writeVideo(v,out);
     videoPlayer(out);
     pause(0.1);
end

close(v)
release(videoPlayer);

%https://uk.mathworks.com/help/vision/ug/using-kalman-filter-for-object-tracking.html

function trackSingleObject(param)
  % Create utilities used for reading video, detecting moving objects,
  % and displaying the results.
  utilities = createUtilities(param);

  isTrackInitialized = false;
  while hasFrame(utilities.videoReader)
    frame = readFrame(utilities.videoReader);

    % Detect the ball.
    [detectedLocation, isObjectDetected] = detectObject(frame);

    if ~isTrackInitialized
      if isObjectDetected
        % Initialize a track by creating a Kalman filter when the ball is
        % detected for the first time.
        initialLocation = computeInitialLocation(param, detectedLocation);
        kalmanFilter = configureKalmanFilter(param.motionModel, ...
          initialLocation, param.initialEstimateError, ...
          param.motionNoise, param.measurementNoise);

        isTrackInitialized = true;
        trackedLocation = correct(kalmanFilter, detectedLocation);
        label = 'Initial';
      else
        trackedLocation = [];
        label = '';
      end

    else
      % Use the Kalman filter to track the ball.
      if isObjectDetected % The ball was detected.
        % Reduce the measurement noise by calling predict followed by
        % correct.
        predict(kalmanFilter);
        trackedLocation = correct(kalmanFilter, detectedLocation);
        label = 'Corrected';
      else % The ball was missing.
        % Predict the ball's location.
        trackedLocation = predict(kalmanFilter);
        label = 'Predicted';
      end
    end

    annotateTrackedObject();
  end % while

  showTrajectory();
end

