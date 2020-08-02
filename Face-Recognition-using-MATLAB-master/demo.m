clc; 
clear;
close all;
objects = imaqfind
delete(objects)

facedatabase = imageSet('data','recursive');        
                
[training,test] = partition(facedatabase,[0.8 0.2]);
trainingFeatures = zeros(size(training,2)*training(1).Count,4680);
featureCount = 1;

for i=1:size(training,2)
    for j= 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),1));
        trainingLabel{featureCount} = training(i).Description ;
        featureCount = featureCount +1;
    end
    personIndex{i}= training(i).Description;

end

faceDetector = vision.CascadeObjectDetector();
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

cam = videoinput('winvideo', 1);

videoFrame = getsnapshot(cam);
frameSize = size(videoFrame);

videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop

    videoFrame = getsnapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;

    if numPts < 10
        bbox = faceDetector.step(videoFrameGray);

        if ~isempty(bbox)
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

          
            oldPoints = xyPoints;

            bboxPoints = bbox2points(bbox(1, :));

            bboxPolygon = reshape(bboxPoints', 1, []);

            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

        end

    else
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            bboxPoints = transformPointsForward(xform, bboxPoints);
            bboxPolygon = reshape(bboxPoints', 1, []);

            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end

    end
    step(videoPlayer, videoFrame);

    runLoop = isOpen(videoPlayer);
    
    try
        faceClassifier = fitcecoc(trainingFeatures,trainingLabel);
        FDetect = vision.CascadeObjectDetector('MergeThreshold',8);
        capturedImage = getsnapshot(cam);
        FaceDetector = vision.CascadeObjectDetector('MergeThreshold',8);
        BBOX = step(FaceDetector, capturedImage);
        B = insertObjectAnnotation(capturedImage, 'rectangle', BBOX, 'Face');

        RGB = insertShape(capturedImage, 'Rectangle', BBOX, 'LineWidth', 5);
        
        s1 = int2str(1);
       s2 = '.jpg';

       s3 = '.pgm';


       s = strcat(s1,s2);

       rgbImage = capturedImage;
       imwrite(rgbImage,s);
       A = imread(s);

       FaceDetector = vision.CascadeObjectDetector();
       BBOX = step(FaceDetector, A);

       Face=imcrop(A,BBOX);

       grayImage = rgb2gray(Face); 
       J = imresize(grayImage,[112 92]);
        
    queryImage = J;
    queryFeatures = extractHOGFeatures(queryImage);
    personLabel = predict(faceClassifier,queryFeatures);

    booleanIndex = strcmp(personLabel,personIndex);
    integerIndex = find(booleanIndex);
    
    if integerIndex == 1
        figure(2);
        
        imshow(read(training(integerIndex),1));title('Evans Raja');
    end
    catch ex
    end
end

clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);