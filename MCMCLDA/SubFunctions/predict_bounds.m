%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Returns the object bounds predicted by a poselet given the poselet
%%% bounds and trained parameters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function bounds = predict_bounds(poselet_bounds, poselet2bounds)
    % Given part hits generates a list of torso predictions for each image

    scale = min(poselet_bounds(3:4)); % The poselet normalized coords go from -0.5 to 0.5 along the shorter dimension
    image2poselet_ctr = poselet_bounds(1:2)+poselet_bounds(3:4)/2;

    scaled_bounds = poselet2bounds.obj_bounds * scale;
    poselet2bounds_ctr = scaled_bounds(1:2) + scaled_bounds(3:4)/2;
    bounds_dims = scaled_bounds(3:4);

    image2bounds_ctr = image2poselet_ctr + poselet2bounds_ctr;        
    bounds = [image2bounds_ctr - bounds_dims/2 bounds_dims];
end