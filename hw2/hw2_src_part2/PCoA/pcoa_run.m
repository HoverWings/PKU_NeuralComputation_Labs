clear, close all

%% extract patches and integrate them into ONE big matrix X
X = zeros(64,5000);
for i = 1:10
    name = sprintf('im.%d.tif', i);
    img = imread(name);
    X(:, 500*(i-1)+ 1 : 500*i) = extract_patches(img, 8, 500);
end

% Create a Bray-Curtis dissimilarity matrix among observations:
dis = f_dis(X,'bc');
 
% Perform Principal Coordinates Analysis, show diagnostic plots:
pcoa = f_pcoa(dis,1);

% Create ordination diagram and biplot vectors (use 'weighted' vectors
% because the data represents species abundance data):
[h,vec] = f_pcoaPlot(pcoa,site_labels,spiders,spiders_labels,0,'none',1);

% Plot vectors for the 3 most important taxa for each axis:
[null,idx_I]  = sortrows(abs(vec),-1); % sort rows descending by column 1
[null,idx_II] = sortrows(abs(vec),-2); % sort rows descending by column 2
idx = f_unique([idx_I(1:3);idx_II(1:3)]);
idx = idx(1:3);
f_pcoaPlot(pcoa,site_labels,spiders(:,idx),spiders_labels(idx),0,'none',1);
