function Vfinal = mytest(learning_rule,numPatterns,sizes,custom_load,updatemod,updatepara,inputmod,inputpara, lrmodifier)


% e.g. mytest('oja', 5, 50, ["bike.jpg", "cake.jpg", "coat.jpg", "dog.jpg", "house.jpg"], 'all', [10, 2], 'corrupt', [4, 100, 10, 5], 0.1)
% e.g. mytest('hebbian', 20, 50, ["bike.jpg", "bird.jpg", "boat.jpg", "bottle.jpg", "cake.jpg", "car.jpg", "cat.jpg", "chair.jpg", "coat.jpg", "dog.jpg", "elephant.jpg", "face.jpg", "horse.jpg", "house.jpg", "plane.jpg", "shoes.jpg", "plant.jpg", "spider.jpg", "sofa.jpg", "tree.jpg"], 'all', [10, 2], 'corrupt', [17, 100, 10, 5], 0.1)
% e.g. mytest('hebbian', 5, 50, ["bike.jpg", "cake.jpg", "coat.jpg", "dog.jpg", "house.jpg"], 'all', [10, 2], 'corrupt', [4, 100, 10, 5], 0.1)
% e.g mytest('oja', 5, 50, ["bike.jpg", "cake.jpg", "chair.jpg", "dog.jpg", "house.jpg"], 'all', [10, 2], 'partial', [1, 25, 10, 10], 0.1)

%parameter definitions.
% learning rule = 'hebbian' (outer product), 'storkey', 'oja'
% 2 = number of patterns to store, the routine customize load decides the
% sequence and collection of the images to store.
% updatemod =  "random", "all", "checkerboard" were options, now always use "all"
% updatepara = [x,y] x = number of iterations, y= number of checkpoints to print out intermediate result
% inputmod = types of  preprocessing the input to be tested, "corrupt", "partial", "full". 
% inputpara = is a list   [x,y,z,w] x= which image to be tested, e.g. 1, or 2. y = specify the degree of noise corruption
%                    y= 20 is adding 20 points, z, w useless for corrupt.
%                   for "partial", x =which image, y = size of the patch visible, (z, w) is the upper left corner coordinate of the visible patch
%                   for "full", just need to provide x, the other numbers are not used or may not needed.


num_iterations=updatepara(1);
checkpoint_number=updatepara(2);
numNeurons = sizes*sizes;

%load mypattern.mat
if length(custom_load)==numPatterns
    mypatterns = zeros(sizes*sizes,1,numPatterns);
    for i =1:numPatterns
        mypatterns(:,:,i)=load_image_by_name(custom_load(i),sizes);
    end
else mypatterns=load_natural_image(numPatterns,sizes);
end

%aa=load_image_by_name("images/bike.jpg",size);
fhi=figure('Units','pixels','Position', [100 100 numPatterns*300 300]);
colormap gray;
for i=1:numPatterns
    subplot(1,numPatterns,i);
    imagesc(reshape(mypatterns(:,:,i),sizes,sizes));
end

T = zeros(numNeurons);
disp(learning_rule);
if strcmp(learning_rule,'hebbian')
    for alpha=1:numPatterns
        data = reshape(mypatterns(:, :, alpha),1,numNeurons);
        T = T + data'*data;
    end
    T = T./numPatterns;   % this normalized by the number of patterns.
elseif strcmp(learning_rule,'storkey')
    for alpha=1:numPatterns
        disp(alpha);
        tmp=T;
        h=T*mypatterns(:,1,alpha);
        for i=1:numNeurons
            for j=1:numNeurons
                hij=h(i)-T(i,i)*mypatterns(i,1,alpha)-T(i,j)*mypatterns(j,1,alpha);
                hji=h(j)-T(j,j)*mypatterns(j,1,alpha)-T(j,i)*mypatterns(i,1,alpha);
                T(i,j)=tmp(i,j)+(mypatterns(i,1,alpha)*mypatterns(j,1,alpha)-mypatterns(i,1,alpha)*hji-mypatterns(j,1,alpha)*hij)/numNeurons;     
            end
        end
    end
elseif strcmp(learning_rule,'oja')
    T=ones(numNeurons);
    T = T./numNeurons;    
    lr=(0.7/numNeurons)*lrmodifier;
    for epoch=1:100
        for iter=1:numPatterns
            xcur = mypatterns(:,:,1:iter);
            xcur = reshape(xcur,2500,iter);
            ycur = xcur'*T';
            deltaT = lr*(xcur - T*ycur')*ycur;
            T=T+deltaT;
        end
    end
    for iter = 1:numPatterns
        for epoch = 1:100
            xcur = mypatterns(:,:,1:iter);
            xcur = reshape(xcur,2500,iter);
            ycur = xcur'*T';
            deltaT = lr*(xcur - T*ycur')*ycur;
            T=T+deltaT;
        end
    end
end
% 

fhi=figure();
colormap gray;
imagesc(T);



Vss=mypatterns(:,:,1);
if strcmp(inputmod,'corrupt')
    Vss =corrupt(mypatterns(:,:,inputpara(1)),inputpara(2));
elseif strcmp(inputmod,'partial')
    Vss =partial_image(mypatterns(:,:,inputpara(1)),inputpara(2),inputpara(3),inputpara(4));
elseif strcmp(inputmod,'full')
    Vss=mypatterns(:,:,inputpara(1));
end

fhi=figure();
colormap gray;
imagesc(reshape(Vss,sizes,sizes));


Vfinal = runHopnet(T,num_iterations,checkpoint_number,Vss,updatemod);

fhi=figure();
colormap gray;
imagesc(reshape(Vfinal,sizes,sizes));



% for i =1:length(custom_load)
%     Vss =corrupt(mypatterns(:,:,inputpara(1)),inputpara(2));
% end

end












