clear, close all

%% extract patches and integrate them into ONE big matrix X
X = zeros(64,5000);
for i = 1:10
    name = sprintf('im.%d.tif', i);
    img = imread(name);
    X(:, 500*(i-1)+ 1 : 500*i) = extract_patches(img, 8, 500);
end

%% compute the covariance matrix  &&  compute the eigens
C = cov(X', 1);
[U, S, V] = svd(C);

%%
Eigs = diag(S);
x = [0:0.2:0.8 0:0.2:0.8]';
y = [0.5*ones(1,5) zeros(1,5)]';
position = [x y 0.18*ones(10,1) 0.5*ones(10,1)];

figure
for j = 1:10
    PC = reshape( U(:,j),8,8);
    PCLarge = imresize(PC,8);
    subplot(2,5,j)
    imshow(PCLarge, [])
    eigV = sprintf('%.2f', Eigs(j));
    title(eigV,'fontsize',10)
    set(gca,'position', position(j,:));
end

%% plot 64 PCs
figure
for j = 1:64
    PC = reshape( U(:,j),8,8);
    PCLarge = imresize(PC,8);
    subplot(8,8,j)
    imshow(PCLarge, [])
end


%% plot 64 eigenvalues in descending order
figure,  plot(Eigs,'r','LineWidth',2)
xlabel('index of ranked eigenvectors','FontSize', 14);
ylabel('eigenvalues','FontSize', 14);

%%
SumEigs = sum(Eigs);
percent = 0;
k = 0;
while percent < 0.99  % change this percent for different levels of variance capture
    k = k+1;
    percent = percent + Eigs(k)/SumEigs;
end
k

%% synthesize image based on principle components
img = imread('im11.tif');
imshow(img,[])
NewImage = zeros(480,480);
k = 0;
PCNum = 4; %% change number of principle components here
Error = 0;
Coe = zeros(3600,PCNum);
for i = 1:8:480
    for j = 1:8:480
        k = k+1;
        block = double(reshape(img(i:i+7,j:j+7),64,1)); 
        coe = block' * U(:,1:PCNum);
        Coe(k,:) = coe;
        synthesize = coe * (U(:,1:PCNum)');
        NewImage(i:i+7,j:j+7) = reshape(synthesize',8,8);
        Error = Error + sum((synthesize' - block).^2)/sum(block.^2);
    end
end
figure
imshow(NewImage,[])
Error = Error/3600  % PercentError


% These codes check whether the principal component (neurons)'s responses to the images are correlated or not. Should they be or not? 
%% plot histograms
if PCNum == 10
    figure
    for i = 1:10
        STD(i) = std(Coe(:,i),1);
        subplot(2,5,i)
        hist(Coe(:,i),40)
        sTD = sprintf('%.2f', STD(i));
        title(strcat('STD:', sTD))
    end
    corrcoef(Eigs(1:10),STD)
end


%%

Patches = X(:,randsample(1:5000,100));
Coe123 = Patches' * U(:,[1,2,3]);
corr = corrcoef(Coe123)









