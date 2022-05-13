function [zernikecoeff_P,Waberration1,PSFzernike, sigmaxy,RRSE,MAPE,z_pos]=zernikefitBeadstack_all(data,data_avg,p,axzernike,axpupil,beadspos_single,aberrations_avg0,aberrations_avg1,z0_shift,sigmaxy_avg) %modify by shiwei 2020/08/20
% function [zernikecoeff_P0 ,zernikecoeff_P, sigmaxy,PSFzernike,rmse0,mape0,rmse1,mape1,rmse_avg0,mape_avg0,rmse_avg1,mape_avg1]=zernikefitBeadstack_all(data,data_avg,p,axzernike,axpupil,beadspos_single) 
% function [paraSim,PSFzernike]=zernikefitBeadstack(data,p,axzernike,axpupil,axmode)

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
% paraFit.aberrations = [2,-2,0.0; 2,2,0.0; 3,-1,0.0; 3,1,0.0; 4,0,0.0; 3,-3,0.0; 3,3,0.0; 4,-2,0.0; 4,2,0.0; 5,-1,0.0; 5,1,0.0; 6,0,0.0; 4,-4,0.0; 4,4,0.0;  5,-3,0.0; 5,3,0.0;  6,-2,0.0; 6,2,0.0; 7,1,0.0; 7,-1,0.0; 8,0,0.0];
% paraFit.aberrations(:,3) =  paraFit.aberrations(:,3)*paraFit.lambda;
paraFit.aberrations  = aberrations_avg1;

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
%     z0(i) = 0;
    z0(i) = z0_shift;  %modify by shiwei 2021/08/24
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
if  all(beadspos_single(1:2) ==[1043,1046]) || all(beadspos_single(1:2) == [552,609])
% if  all(beadspos_single(1:2) ==[197,109])
    beadspos_single;
end


% first fit
% [P0,model0,err0] = MLE_FitAbberation_Final_GPU_float(data_double,thetainit_d,paraFitCell,shared_d,0.1,0);
% % rmse.rmse0 = sqrt(mean(sum(sum(( (model0-data)./max(max(model0)) ).^2))/size(data,1)/size(data,2)));  %add by shiwei 2021/06/01 from fushuang
% 
% % rmse.rmse0 = sqrt(mean(mean(mean((model0-data).^2))));
% model_err0 = model0-data;
% RRSE.RRSE0 = norm(model_err0(:),2) / norm(data(:),2);
% % rmse.rmse0 =sqrt(mean(sum(sum(( (model0-data)./max(max(model0)) ).^2))));
% MAPE.MAPE0 = mean(sum(abs(model_err0),[1,2])./sum(data,[1,2]))*100;%add by shiwei 2021/06/05 from fushuang
% 
% % rmse.rmse_avg0 = sqrt(mean(sum(sum(( (model0-data_avg)./max(max(model0)) ).^2))/size(data_avg,1)/size(data_avg,2)));  %add by shiwei 2021/06/08
% % rmse.rmse_avg0 =sqrt(mean(mean(mean((model0-data_avg).^2))));
% model_avg_err0 = model0-data_avg;
% RRSE.RRSE_avg0 = norm(model_avg_err0(:),2) / norm(data_avg(:),2);
% % rmse.rmse0 =sqrt(mean(sum(sum(( (model0-data_avg)./max(max(model0)) ).^2))));
% MAPE.MAPE_avg0 = mean(sum(abs(model_avg_err0),[1,2])./sum(data_avg,[1,2]))*100;
% % rmse.std_aberrations0 =sqrt(mean(((P0(1:21)-aberrations_avg0(:,3))./size(aberrations_avg0,1)).^2));
% RRSE.STD_aberrations0 =sqrt(mean((P0(1:21)-aberrations_avg0(:,3)).^2));
% MAPE.MAPE_aberrations0 = mean(abs(P0(1:21)-aberrations_avg0(:,3))./abs(aberrations_avg0(:,3)))*100;
% 
% % toc
% 
% end
% 
% % tic
% % [P0_cpu,model0_cpu,err0_cpu] = MLE_FitAbberation_Final_FixPara_Regu_sigma(double(data),thetainit,paraFit,shared,0.1,0);
% % % [P0,model0,err0] = MLE_FitAbberation_Final_FixPara_Regu_sigma(double(data),thetainit,paraFit,shared,0.1,0);
% % toc
% 
% beadStackPos = paraFit.objStageStack+P0(24);
% z_pos(1) = P0(24);

%% start original sigma computer 

% [val,ind]=min(abs(beadStackPos));
% 
% measuredPSF = data(:,:,ind);
% % measuredPSF = padarray(measuredPSF,[(64-size(measuredPSF,1))/2 (64-size(measuredPSF,2))/2],0,'both');
% modelPSF = model0(:,:,ind);
% % modelPSF = padarray(modelPSF,[(64-size(modelPSF,1))/2 (64-size(modelPSF,2))/2],0,'both');
% 
% mOTF=fftshift(ifft2(measuredPSF));% measured OTF
% rOTF=fftshift(ifft2(modelPSF)); % phase retrieved OTF
% tmp=abs(mOTF)./abs(rOTF);
% R = 5;
% tmp1 = tmp((Npixel+1)/2-R:(Npixel+1)/2+R,(Npixel+1)/2-R:(Npixel+1)/2+R);
% % [I,sigmax,sigmay,bg]=GaussRfit(obj,ratio);
% % x = -64/2:64/2-1;
% x =1:2*R+1;
% % Ix = mean(tmp1,1);
% % Iy = mean(tmp1,2);
% Ix = roundn(mean(tmp1,1),-4);  %modify by shiwei 2021/08/18 solve problem about "NaN computed by model function, fitting cannot continue."
% Iy = roundn(mean(tmp1,2),-4);
% 
% fx = fit(x',Ix','gauss1');
% sigmax = fx.c1/sqrt(2);
% sigmax = max(5,sigmax);
% fy = fit(x',Iy,'gauss1');
% sigmay = fy.c1/sqrt(2);
% sigmay = max(5,sigmay);
% % I = (fx.a1 + fy.a1)/2;
% % 
% % Ixf = feval(fx,x);
% % Iyf = feval(fy,x);
% % % figure;plot(x,Ix,'bo',x,Ixf,'r-')
% % % figure;plot(x,Iy,'bo',x,Iyf,'r-')
% % bg = 0;
% [xx,yy]=meshgrid(-(Npixel-1)/2:(Npixel-1)/2,-(Npixel-1)/2:(Npixel-1)/2);
% X=abs(xx);
% Y=abs(yy);
% fit_im=exp(-X.^2./2./sigmax^2).*exp(-Y.^2./2./sigmay^2);
% tempGauss = abs(fftshift(ifft2(fit_im)));
% tempGauss1 = tempGauss((Npixel+1)/2-2:(Npixel+1)/2+2,(Npixel+1)/2-2:(Npixel+1)/2+2);
% tempGauss1 = tempGauss1/sum(tempGauss1(:));
% % thetainit(1:21)=P0(1:21);

%% end original sigma computer 


%%
% calculate the psf gauss directly using Isigma   
%start add by fushuang 2021/08/21
% I_sigmax=1/(2*pi*otf_sigmax)*Npixel_otf/ds_step;
% I_sigmay=1/(2*pi*otf_sigmay)*Npixel_otf/ds_step;
I_sigmax=sigmaxy_avg(1);
I_sigmay=sigmaxy_avg(2);
[x1,y1]=meshgrid( - 2 : 2 );
gauss_psf=exp(-x1.^2./2./I_sigmax^2).*exp(-y1.^2./2./I_sigmay^2);
gauss_psf = gauss_psf/sum(gauss_psf,'all');
% figure;mesh(gauss_psf);title('psf gauss calculated by Isigma');
 tempGauss1=gauss_psf;
%end  add by fushuang 2021/08/21
%%

% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% otf_sigmax=sigmax;
% otf_sigmay=sigmay;
% 
% Npixel_otf=131;
% ds_step = (Npixel_otf-1) / (27-1);
% % calculate the psf gauss directly using Isigma
% I_sigmax=1/(2*pi*otf_sigmax)*Npixel_otf/ds_step;
% I_sigmay=1/(2*pi*otf_sigmay)*Npixel_otf/ds_step;
% [x1,y1]=meshgrid( - 2 : 2 );
% gauss_psf=exp(-x1.^2./2./I_sigmax^2).*exp(-y1.^2./2./I_sigmay^2);
% gauss_psf = gauss_psf/sum(gauss_psf,'all');
% figure;mesh(gauss_psf);title('psf gauss calculated by Isigma');
% tempGauss1=gauss_psf;
% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%

if 1
    
%    tic
%[P_cDouble2,model_cDouble2,err_cDouble2] = MLE_FitAbberation_Final_CDouble2(data_double,thetainit_d,paraFitCell,shared_d,0.1,tempGauss1);
%[P_cDouble2,model_cDouble2,err_cDouble2] = MLE_FitAbberation_Final_GPU_Double(data_double,thetainit_d,paraFitCell,shared_d,0.1,tempGauss1);
[P,model,err] = MLE_FitAbberation_Final_GPU_float(data_double,thetainit_d,paraFitCell,shared_d,0.1,tempGauss1);
% toc 

% rmse.rmse1 = sqrt(mean(sum(sum(( (model-data)./max(max(model)) ).^2))/size(data,1)/size(data,2)));  %add by shiwei 2021/06/01 from fushuang
% rmse.rmse1 = sqrt(mean(mean(mean((model-data).^2))));
z_pos = P(24);
model_err1= model-data;
RRSE.RRSE1= norm(model_err1(:),2) / norm(data(:),2);

% rmse.rmse1 =sqrt(mean(sum(sum(( (model-data)./max(max(model)) ).^2))));

MAPE.MAPE1 = mean(sum(abs(model_err1),[1,2])./sum(data,[1,2]))*100;%add by shiwei 2021/06/05 from fushuang
model_avg_err1= model-data_avg;

RRSE.RRSE_avg1= norm(model_avg_err1(:),2) / norm(data_avg(:),2);
% rmse.rmse_avg1 = sqrt(mean(mean(mean((model-data_avg).^2))));
% rmse.rmse_avg1 = sqrt(mean(sum(sum(( (model-data_avg)./max(max(model)) ).^2))));
% rmse.rmse_avg1 = sqrt(mean(sum(sum(( (model-data_avg)./max(max(model)) ).^2))/size(data_avg,1)/size(data_avg,2)));  %add by shiwei 2021/06/08
MAPE.MAPE_avg1 = mean(sum(abs(model_avg_err1),[1,2])./sum(data_avg,[1,2]))*100;%add by shiwei 2021/06/08


RRSE.STD_aberrations1 =sqrt(mean((P(1:21)-aberrations_avg0(:,3)).^2));
MAPE.MAPE_aberrations1 = mean(abs(P(1:21)-aberrations_avg0(:,3))./abs(aberrations_avg0(:,3)))*100;

end

% tic
%[P,model,err] = MLE_FitAbberation_Final_FixPara_Regu_sigma2(double(data),thetainit,paraFit,shared,0.1,tempGauss1);
% toc
%[P,model,err] = MLE_FitAbberation_Final_FixPara_Regu_sigma(double(data),thetainit,paraFit,shared,0.1,0);


% err0
% err
% % P = P0;
% % model = model0;
% % err = err0;

paraSim=paraFit;
% paraSim.aberrations(:,3)=P0(1:21);
% zernikecoeff_P0 = paraSim.aberrations;

paraSim.aberrations(:,3)=P(1:21);

%% start original sigma computer 
% paraSim.OTFsigmax = sigmax;
% paraSim.OTFsigmay = sigmay;
% paraSim.Imgsigmax = 1/sigmax/(2*pi/size(measuredPSF,1));
% paraSim.Imgsigmay = 1/sigmay/(2*pi/size(measuredPSF,2));
% sigmaxy=[paraSim.Imgsigmax,paraSim.Imgsigmay];%add by shiwei 20210518
%% end original sigma computer 

% sigmaxy=[sigmax,sigmay];%add by shiwei 20210518

sigmaxy=[I_sigmax,I_sigmay];

% aberr_perc0 = sum(abs(model0-data),[1,2]) ./ sum(data,[1,2]);%add by shiwei 20210601 from fushuang
% modelP=vectorPSFsimple(paraSim);

% imx(cat(1,data,model,data-model),'Parent',axzernike)   %modify by shiwei
% 2020/08/14

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
% figure,imshow(Waberration1);
%% start modify by shiwei 2020/08/25
% if 1 == beadspos_single(3)
%     figure
%     imx(cat(1,data,model,data-model))
%     figure_title = title(['(x y ) coordinates is ','(',num2str(beadspos_single(1:2)),')']);
%     set(figure_title,'FontName','time','FontSize',17,'LineWidth',3,'FontWeight','bold');
% end

% % start mark show aberration
% figure,bar([P0(1:21),P(1:21)]);
% 
% for k=size(orders1):-1:1
%     axn{k}=[num2str(orders1(k,1)) ',' num2str(orders1(k,2))];
% end
% set(gca,'XTick',1:length(axn));
% set(gca,'XTickLabel',axn);
% 
% legend( {'Original','OTF Rescale'})
% title(['(x y ) coordinates is ','(',num2str(beadspos_single),')']);
% 
% figure,imagesc(Waberration1);
% axis('equal')
% axis('tight')
% colorbar
% title(['(x y index) coordinates is ','(',num2str(beadspos_single),')']);
% % end mark show aberration

%% »­Í¼ÓÃ
% figure,b = bar(P(1:21),'FaceColor','r','EdgeColor','r');
% % ylim([-50:100])
% set(gca,'YLim',[-60 100]);
% set(gca,'YTick',[-60:20:100]);
% % b.FaceColor = [0.6350 0.0780 0.1840];
% % b.EdgeColor = [0.6350 0.0780 0.1840];
% b.FaceColor = 'flat';
% b.EdgeColor = 'flat';
% b.CData(2,:) = [0.8500 0.3250 0.0980];
% b.CData(1,:) = [0.9290 0.6940 0.1250];  
% for k =3:21
%  b.CData(k,:) = [0.9290 0.6940 0.1250];   
% end
%     
% for k=size(orders1):-1:1
%     axn{k}=[num2str(orders1(k,1)) ',' num2str(orders1(k,2))];
% end
% set(gca,'XTick',1:length(axn));
% set(gca,'XTickLabel',axn);
% xtickangle(45)
% % legend( {'OTF Rescale'})
% title(['(x y ) coordinates is ','(',num2str(beadspos_single(1:2)-88),')']);
% ylabel('nm') ,xlabel('zernike mode')
% 
% set(gca,'FontName','time','FontSize',12,'FontWeight','bold');
%%

zernikecoeff_P = paraSim.aberrations;

% paraSim.aberrations(:,3)=P0(1:21);
% zernikecoeff_P0 = paraSim.aberrations;
% zernikecoeff_P0=P0(1:21);


% imagesc(axpupil,Waberration1)
% axis(axpupil,'equal')
% axis(axpupil,'tight')
% % ax1=axzernike{1};
% % imagesc(ax1,[Waberration0,Waberration1]);
% 
% 
% bar(axmode,[P0(1:21),P(1:21)])
% legend(axmode,{'Original','OTF Rescale'})
% 
% for k=size(orders1):-1:1
%     axn{k}=[num2str(orders1(k,1)) ',' num2str(orders1(k,2))];
% end
% axmode.XTick=1:length(axn);
% axmode.XTickLabel=axn;
%% end modify by shiwei 2020/08/25

PSFzernike=model;
end






