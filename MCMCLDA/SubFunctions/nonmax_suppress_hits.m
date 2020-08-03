%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% Removes any hits that overlap intersection-over-union by more than
%%% match_thresh with other hits with higher scores
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hits,valid] = nonmax_suppress_hits(hits,match_thresh)
if nargin<2
    match_thresh=0.5;
end

poselet_ids = unique(hits.poselet_id)';
valid = false(hits.size,1);
for p=poselet_ids
    phits=find(hits.poselet_id==p);
    valid(phits(get_nonoverlapped_bounds(hits.bounds(:,phits),hits.score(phits),inf,match_thresh)))=true;
end
hits = hits.select(valid);
    
end