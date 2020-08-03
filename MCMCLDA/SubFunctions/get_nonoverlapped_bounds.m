%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Given a set of possibly overlapping bounds and associated scores, returns
%%% the list of bounds that are not overlapped by more than match_thresh by other
%%% bounds with higher score

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function idx = get_nonoverlapped_bounds(bounds,scores,max_hits,match_thresh)

if nargin<4
   match_thresh=0.5; 
end
[srt,srtd] = sort(scores,'descend');

ibounds = [srtd'; bounds(:,srtd)];

i=1;
while i<size(ibounds,2) && i<max_hits
    suppressed = [zeros(i,1); bounds_overlap(ibounds(2:5,i), ibounds(2:5,(i+1):end))>=match_thresh];
    ibounds = ibounds(:,~suppressed);
    i=i+1;
end

idx=ibounds(1,:);

end
