function [paraSim,PSFzernike,aberrations_avg0,aberrations_avg1,RRSE,MAPE,sigmaxy_avg]=zernikefitBeadstack(data,p,axzernike,axpupil,axmode)
Npixel = size(data,1);
Nz = size(data,3);
paraFit=p;
paraFit.sizeX = size(data,1);
paraFit.sizeY = size(data,2);
paraFit.sizeZ = size(data,3);
% aberrations (Zernike orders [n1,m1,A1,n2,m2,A2,...] with n1,n2,... the
% radial orders, m1,m2,... the azimuthal orders, and A1,A2,... the Zernike
% coefficients in lambda rms, so 0.072 means diffraction limit)
% parameters.aberrations = [1,1,0.0; 1,-1,-0.0; 2,0,-0.0; 4,0,0.0; 2,-2,0.0; 2,2,0.0; 4,-2,0.0];
paraFit.aberrations = [2,-2,0.0; 2,2,0.0; 3,-1,0.0; 3,1,0.0; 4,0,0.0; 3,-3,0.0; 3,3,0.0; 4,-2,0.0; 4,2,0.0; 5,-1,0.0; 5,1,0.0; 6,0,0.0; 4,-4,0.0; 4,4,0.0;  5,-3,0.0; 5,3,0.0;  6,-2,0.0; 6,2,0.0; 7,1,0.0; 7,-1,0.0; 8,0,0.0];
paraFit.aberrations(:,3) =  paraFit.aberrations(:,3)*paraFit.lambda;

numAberrations = size(paraFit.aberrations,1);
shared = [ones(1,numAberrations) 1 1 1 p.sharedIB p.sharedIB ];  % 1 is shared parameters between z slices, 0 is free parameters between z slices, only consider  [x, y, z, I, bg]

sumShared = sum(shared);
numparams = 26*Nz-sumShared*(Nz-1);


thetainit = zeros(numparams,1);

bg = zeros(1,Nz);
Nph = zeros(1,Nz);
x0 = zeros(1,Nz);
y0 = zeros(1,Nz);
z0 = zeros(1,Nz);

% center of mass with nm unit
ImageSizex = paraFit.pixelSizeX*Npixel/2;
ImageSizey = paraFit.pixelSizeY*Npixel/2;

DxImage = 2*ImageSizex/paraFit.sizeX;
DyImage = 2*ImageSizey/paraFit.sizeY;
ximagelin = -ImageSizex+DxImage/2:DxImage:ImageSizex;
yimagelin = -ImageSizey+DyImage/2:DyImage:ImageSizey;
[YImage,XImage] = meshgrid(yimagelin,ximagelin);
for i = 1:Nz
    dTemp = data(:,:,i);
    bg(i) = min(dTemp(:));
    bg(i) = max(bg(i),1);
    Nph(i) = sum(sum(dTemp-bg(i)));
    x0(i) = sum(sum(XImage.*dTemp))/Nph(i);
    y0(i) = sum(sum(YImage.*dTemp))/Nph(i);
    z0(i) = 0;
end


allTheta = zeros(numAberrations+5,Nz);
allTheta(numAberrations+1,:)=x0';
allTheta(numAberrations+2,:)=y0';
allTheta(numAberrations+3,:)=z0';
allTheta(numAberrations+4,:)=Nph';
allTheta(numAberrations+5,:)=bg';
allTheta(1:numAberrations,:) = repmat(paraFit.aberrations(:,3),[1 Nz]);

% for 
map = zeros(numparams,3);
n=1;
for i = 1:numAberrations+5
    if shared(i)==1
        map(n,1)= 1;
        map(n,2)=i;
        map(n,3)=0;
        n = n+1;
    elseif shared(i)==0
        for j = 1:Nz
            map(n,1)=0;
            map(n,2)=i;
            map(n,3)=j;
            n = n+1;
        end
    end
end



for i = 1:numparams
    if map(i,1)==1
        thetainit(i)= mean(allTheta(map(i,2),:));
    elseif map(i,1)==0
         thetainit(i) = allTheta(map(i,2),map(i,3));
    end
end

% we assume that parameters for zernike coefficiens are always linked
zernikecoefsmax = 0.25*paraFit.lambda*ones(numAberrations,1);
paraFit.maxJump = [zernikecoefsmax',paraFit.pixelSizeX*ones(1,max(Nz*double(shared(numAberrations+1)==0),1)),paraFit.pixelSizeY*ones(1,max(Nz*double(shared(numAberrations+2)==0),1)),500*ones(1,max(Nz*double(shared(numAberrations+3)==0),1)),2*max(Nph(:)).*ones(1,max(Nz*double(shared(numAberrations+4)==0),1)),100*ones(1,max(Nz*double(shared(numAberrations+5)==0),1))];


%% fit data
paraFit.numparams = numparams;
paraFit.numAberrations = numAberrations;
paraFit.zemitStack = zeros(size(data,3),1); % move emitter
% paraFit.objStageStack = 600:-30:600-30*40; %move objStage
zmax=(paraFit.sizeZ-1)*paraFit.dz/2;
paraFit.objStageStack=zmax:-paraFit.dz:-zmax;


paraFit.ztype = 'stage';
paraFit.map = map;
paraFit.Nitermax = paraFit.iterations;
% [P,model,err] = MLE_FitAbberation_Final(data,thetainit,paraFit,shared);


if 1
parC.NA = paraFit.NA;
parC.refmed = paraFit.refmed;
parC.refcov = paraFit.refcov;
parC.refimm = paraFit.refimm;
parC.lambda = paraFit.lambda;
parC.zemit0 = paraFit.zemit0;
% parC.zemit0 = 0;
parC.objStage0 = paraFit.objStage0;
parC.pixelSizeX = paraFit.pixelSizeX;
parC.pixelSizeY = paraFit.pixelSizeY;
parC.Npupil = paraFit.Npupil;
parC.sizeX = paraFit.sizeX;
parC.sizeY = paraFit.sizeY;
parC.sizeZ = paraFit.sizeZ;
parC.aberrations = paraFit.aberrations;
parC.maxJump = paraFit.maxJump;
parC.numparams = paraFit.numparams;
parC.numAberrations = paraFit.numAberrations;
parC.zemitStack = paraFit.zemitStack;
parC.objStageStack = paraFit.objStageStack;
parC.ztype = paraFit.ztype;
parC.map = paraFit.map;
parC.Nitermax = paraFit.Nitermax;
  
paraFitCell= struct2cell(parC);
data_double = double(data.*1);
thetainit_d =thetainit.*1;
shared_d=shared.*1;


% tic
%[P_cDouble,model_cDouble,err_cDouble] = MLE_FitAbberation_Final_CDouble2(data_double,thetainit_d,paraFitCell,shared_d,0.1,0);
% toc

% tic
%[P_cDouble,model_cDouble,err_cDouble] = MLE_FitAbberation_Final_CDouble2(data_double,thetainit_d,paraFitCell,shared_d,0.1,0);
%[P_gpu,model_gpu,err_gpu] = MLE_FitAbberation_Final_GPU_Double(data_double,thetainit_d,paraFitCell,shared_d,0.1,0);
[P0,model0,err0] = MLE_FitAbberation_Final_GPU_float(data_double,thetainit_d,paraFitCell,shared_d,0.1,0);
% [P0_C,model0_C,err0_C] = MLE_FitAbberation_Final_CFloat(data_double,thetainit_d,paraFitCell,shared_d,0.1,0);
% [P0,model0,err0] = MLE_FitAbberation_Final_FixPara_Regu_sigma(data,thetainit,paraFit,shared,0.1,0);
% rmse_avg0 = sqrt(mean(sum(sum(( (model0-data)./max(max(model0)) ).^2))/size(data,1)/size(data,2))); %add by shiwei 2021/06/01 from fushuang
% rmse_avg0 = sqrt(mean(mean(mean((model0-data).^2))));
model_all_avg_err0= model0-data;
RRSE.RRSE_all_avg0= norm(model_all_avg_err0(:),2) / norm(data(:),2);  %norn 先平方后相加再开根号
% rmse_avg0 =sqrt(mean(sum(sum(( (model0-data)./max(max(model0)) ).^2))));
MAPE.MAPE_all_avg0 = mean(sum(abs(model_all_avg_err0),[1,2])./sum(data,[1,2]))*100;%add by shiwei 2021/06/05 from fushuang
aberrations_avg0 =  paraFit.aberrations;
aberrations_avg0(:,3)=P0(1:21);
% [P0_cmp,model0_cmp,err0_cmp] = MLE_FitAbberation_Final_FixPara_Regu_sigma(data,thetainit,paraFit,shared,0.1,0);
toc


end

% tic
%[P0,model0,err0] = MLE_FitAbberation_Final_FixPara_Regu_sigma(double(data),thetainit,paraFit,shared,0.1,0);
% toc

beadStackPos = paraFit.objStageStack+P0(24);

[val,ind]=min(abs(beadStackPos));

measuredPSF = data(:,:,ind);
% measuredPSF = padarray(measuredPSF,[(64-size(measuredPSF,1))/2 (64-size(measuredPSF,2))/2],0,'both');
modelPSF = model0(:,:,ind);
% modelPSF = padarray(modelPSF,[(64-size(modelPSF,1))/2 (64-size(modelPSF,2))/2],0,'both');

mOTF=fftshift(ifft2(measuredPSF));% measured OTF
rOTF=fftshift(ifft2(modelPSF)); % phase retrieved OTF
tmp=abs(mOTF)./abs(rOTF);
R = 5; %speed_test 5
tmp1 = tmp((Npixel+1)/2-R:(Npixel+1)/2+R,(Npixel+1)/2-R:(Npixel+1)/2+R);
% [I,sigmax,sigmay,bg]=GaussRfit(obj,ratio);
% x = -64/2:64/2-1;
x =1:2*R+1;
Ix = mean(tmp1,1);
Iy = mean(tmp1,2);

fx = fit(x',Ix','gauss1');
sigmax = fx.c1/sqrt(2);
sigmax = max(5,sigmax);
fy = fit(x',Iy,'gauss1');
sigmay = fy.c1/sqrt(2);
sigmay = max(5,sigmay);
% I = (fx.a1 + fy.a1)/2;
% 
% Ixf = feval(fx,x);
% Iyf = feval(fy,x);
% % figure;plot(x,Ix,'bo',x,Ixf,'r-')
% % figure;plot(x,Iy,'bo',x,Iyf,'r-')
% bg = 0;


[xx,yy]=meshgrid(-(Npixel-1)/2:(Npixel-1)/2,-(Npixel-1)/2:(Npixel-1)/2);
X=abs(xx);
Y=abs(yy);
fit_im=exp(-X.^2./2./sigmax^2).*exp(-Y.^2./2./sigmay^2);
tempGauss = abs(fftshift(ifft2(fit_im)));
tempGauss1 = tempGauss((Npixel+1)/2-2:(Npixel+1)/2+2,(Npixel+1)/2-2:(Npixel+1)/2+2);
tempGauss1 = tempGauss1/sum(tempGauss1(:));
% thetainit(1:21)=P0(1:21);

if 1
    
%    tic
%[P_cDouble2,model_cDouble2,err_cDouble2] = MLE_FitAbberation_Final_CDouble2(data_double,thetainit_d,paraFitCell,shared_d,0.1,tempGauss1);
%[P_cDouble2,model_cDouble2,err_cDouble2] = MLE_FitAbberation_Final_GPU_Double(data_double,thetainit_d,paraFitCell,shared_d,0.1,tempGauss1);
[P,model,err] = MLE_FitAbberation_Final_GPU_float(data_double,thetainit_d,paraFitCell,shared_d,0.1,tempGauss1);
% [P,model,err] = MLE_FitAbberation_Final_FixPara_Regu_sigma2(data,thetainit,paraFit,shared,0.1,tempGauss1);
model_all_avg_err1= model-data;
RRSE.RRSE_all_avg1= norm(model_all_avg_err1(:),2) / norm(data(:),2);
% rmse_avg0 =sqrt(mean(sum(sum(( (model0-data)./max(max(model0)) ).^2))));
MAPE.MAPE_all_avg1 = mean(sum(abs(model_all_avg_err1),[1,2])./sum(data,[1,2]))*100;%add by shiwei 2021/06/05 from fushuang
aberrations_avg1 =  paraFit.aberrations;
aberrations_avg1(:,3)=P(1:21);

toc 
end
% sigma = tempGauss1;
% sigmadir =strcat(pwd,'\sigma.mat');
% save(sigmadir,'sigma');

% tic
%[P,model,err] = MLE_FitAbberation_Final_FixPara_Regu_sigma2(double(data),thetainit,paraFit,shared,0.1,tempGauss1);
% toc
%[P,model,err] = MLE_FitAbberation_Final_FixPara_Regu_sigma(double(data),thetainit,paraFit,shared,0.1,0);


% err0
% err



% P = P0;
% model = model0;
% err = err0;

paraSim=paraFit;
paraSim.aberrations(:,3)=P(1:21);
% paraSim.OTFsigmax = sigmax;
% paraSim.OTFsigmay = sigmay;

paraSim.OTFsigmax = 0.1;
paraSim.OTFsigmay = 0.1;

paraSim.Imgsigmax = 1/sigmax/(2*pi/size(measuredPSF,1));
paraSim.Imgsigmay = 1/sigmay/(2*pi/size(measuredPSF,2));
sigmaxy_avg=[paraSim.Imgsigmax,paraSim.Imgsigmay];%add by shiwei 20210817
% aberr_perc0 = sum(abs(model0-data),[1,2]) ./ sum(data,[1,2]);%add by shiwei 20210601 from fushuang
% modelP=vectorPSFsimple(paraSim);
imx(cat(1,data,model,data-model),'Parent',axzernike);

PupilSize = 1.0;
Npupil = paraFit.Npupil;
DxyPupil = 2*PupilSize/Npupil;
XYPupil = -PupilSize+DxyPupil/2:DxyPupil:PupilSize;
[YPupil,XPupil] = meshgrid(XYPupil,XYPupil);
ApertureMask = double((XPupil.^2+YPupil.^2)<1.0);
orders = paraFit.aberrations(:,1:2);
Waberration1 = zeros(size(XPupil));
orders1 = paraFit.aberrations(:,1:2);
zernikecoefs1 = P(1:21,1);
normfac = sqrt(2*(orders(:,1)+1)./(1+double(orders(:,2)==0)));
zernikecoefs1 = normfac.*zernikecoefs1;
allzernikes1 = get_zernikefunctions(orders1,XPupil,YPupil);
for j = 1:numel(zernikecoefs1)
  Waberration1 = Waberration1+zernikecoefs1(j)*squeeze(allzernikes1(j,:,:));  
end

Waberration1 = Waberration1.*ApertureMask;
imagesc(axpupil,Waberration1);
axis(axpupil,'equal');
axis(axpupil,'tight');
% ax1=axzernike{1};
% imagesc(ax1,[Waberration0,Waberration1]);


bar(axmode,[P0(1:21),P(1:21)]);
legend(axmode,{'Original','OTF Rescale'});
for k=size(orders1):-1:1
    axn{k}=[num2str(orders1(k,1)) ',' num2str(orders1(k,2))];
end
axmode.XTick=1:length(axn);
axmode.XTickLabel=axn;
axmode.XTickLabelRotation=45;

PSFzernike=model;
% PSFzernike=model0;
end





