

function [yhats,maxscores,cascade_scores] = Each_models_side(img,pyra,pyra_full,models,param,genfeatures,gt_mode)
yhats = {};
k = length(models);
maxscores = -inf(1,k);

%%
[possible_modes,cascade_scores] = cascade_filter_modes(img,param.mode_filter_w,param.mode_filter_alpha);
possible_modes = find(possible_modes);
if nargin == 5
    possible_modes = unique([possible_modes(:); gt_mode]);
end
%% apply all masks, filters in 1 go for all models
nmodes = length(possible_modes);
nfilts = length(models{possible_modes(1)});
mm = models(possible_modes);
mf = [mm{:}];
nlvls = min(length(pyra.feat),length(pyra_full.feat));

rootfilters = {mf(1:nfilts:end).filter_root};
filters = {mf.filter};

unary = cell(nfilts,nmodes,nlvls);
unaryfull = cell(nmodes,nlvls);
tic
for l=1:nlvls
    rl = fconv(pyra.feat{l},filters,1,length(mf));
    unary(:,:,l) = reshape(rl,nfilts,nmodes);
    unaryfull(:,l) = fconv(pyra_full.feat{l}(:,:,[19:27 32]),rootfilters,1,nmodes);
end
fprintf('unary time: %.02f secs for %d modes\n',toc,nmodes)

%% mask shoulder in all scales
if ~genfeatures
    sho = [0.66;0.36];
    shoind = 3;
    maskr = 0.1;
    
    sho = sho-maskr/2;
    for l=1:nlvls
        dims = size(unary{shoind,1,l});
        shopt = sho.*dims([2 1])';
        sho_wh = maskr*dims([2 1])';
        shox = round(shopt(1)-sho_wh(1)):round(shopt(1)+sho_wh(1));
        shoy = round(shopt(2)-sho_wh(2)):round(shopt(2)+sho_wh(2));
        mask = true(size(unary{shoind,1,l}));
        mask(shoy,shox) = false;
        for m=1:nmodes
            unary{shoind,m,l}(mask) = -10;
        end
    end
    
    if 0
        %% debug mask
        dims = size(img(:,:,1));
        shopt = sho.*dims([2 1])';
        sho_wh = maskr*dims([2 1])';
        shox = round(shopt(1)-sho_wh(1)):round(shopt(1)+sho_wh(1));
        shoy = round(shopt(2)-sho_wh(2)):round(shopt(2)+sho_wh(2));
        mask = true(dims);
        mask(shoy,shox) = false;
        imagesc(bsxfun(@times,uint8(img),uint8(mask)))
    end
    0;
end

%%
yhats = cell(k,1);
tic
for ii=1:nmodes
    i = possible_modes(ii);
    u = squeeze(unary(:,ii,:));
    ufull = unaryfull(ii,:);
    [yhats{i},maxscores(i)] = inference_one_model(models{i},u,ufull,pyra,pyra_full,param,genfeatures);
    yhats{i}.mode = i;
end
fprintf('inference time: %.02f secs\n',toc)
[maxval,amax] = max(maxscores);

% break ties randomly
amax = shuffle(find(maxscores==maxval));
amax = amax(1);
yhat = yhats{amax};
yhat.mode = amax;
