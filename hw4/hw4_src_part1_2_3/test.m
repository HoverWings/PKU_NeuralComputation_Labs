% (learning_rule,numPatterns,sizes,custom_load,updatemod,updatepara,inputmod,inputpara, lrmodifier)
learning_rule='hebbian';
numPatterns=10;
sizes=50;
custom_load=["bike.jpg", "bird.jpg", "boat.jpg", "bottle.jpg", "cake.jpg", "car.jpg", "cat.jpg", "chair.jpg", "coat.jpg", "dog.jpg", "elephant.jpg", "face.jpg", "horse.jpg", "house.jpg", "plane.jpg", "shoes.jpg", "plant.jpg", "spider.jpg", "sofa.jpg", "tree.jpg"]
updatemod='all';
updatepara=[10, 2];
inputmod='corrupt';
inputpara=[1, 1, 10, 5];
lrmodifier=0.1;
mytest(learning_rule,numPatterns,sizes,custom_load,updatemod,updatepara,inputmod,inputpara, lrmodifier)
for i= 1:1:10
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
end