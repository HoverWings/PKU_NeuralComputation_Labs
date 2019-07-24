% function: runHopnet.m - simulates a Hopfield network
%
% returns the final network state
% parameters:
% update_method{ 
% all: per iteration, update all pixels simultaneously
% }
% T: weight matrix
% num_iterations: number of iterations
% V0: input image

function Vfinal = runHopnet(T,num_iterations,checkpoint_number,V0,update_method)
checkpoint=num_iterations/checkpoint_number;
image_pixels=checkpoint_number*300;
%fhi=figure('Units','pixels','Position', [100 100 image_pixels 300]);
fhi=figure();
colormap gray;

[N N]=size(T);
sz=sqrt(N);

tmp= V0;

if update_method=="all"
    for t=1:num_iterations
        U = T*tmp;
        tmp=sign(U);
        if mod(t,checkpoint)==0
%             colormap gray;
%             subplot(1,checkpoint_number,t/checkpoint);
            imagesc(reshape(tmp,sz,sz));       
        end
    end
end




Vfinal = tmp;
end