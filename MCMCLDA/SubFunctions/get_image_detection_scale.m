function img_scale=get_image_detection_scale(img_dims)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Returns the scale to which the image will be scaled before poselets are
%%% run on it. We scale up small images and scale down large ones

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global config;

if config.DETECTION_IMG_MIN_NUM_PIX>0 || config.DETECTION_IMG_MAX_NUM_PIX<inf
    img_scale = max(1,sqrt(config.DETECTION_IMG_MIN_NUM_PIX/prod(img_dims)));
    if img_scale==1
        img_scale = min(1, sqrt(config.DETECTION_IMG_MAX_NUM_PIX/prod(img_dims)));
    end
else
    img_scale=1; 
end