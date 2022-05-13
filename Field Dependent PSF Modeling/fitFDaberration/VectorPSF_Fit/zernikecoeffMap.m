function maps_gauss = zernikecoeffMap(p,beads_zrnikeparam,axmap,axdist)
pos = beads_zrnikeparam.pos;
value = beads_zrnikeparam.value_P;
RRSE1 =beads_zrnikeparam.RRSE.RRSE1;
%     sigmaxy_avg = beads_zrnikecoeff.sigmaxy_avg;
std_ref=str2num(beads_zrnikeparam.aberrations_std_set.String);
mapsmooth_set=str2num(beads_zrnikeparam.mapsmooth_set.String);
modelrrse_set=str2num(beads_zrnikeparam.modelrrse_set.String);
localroi_set=str2num(beads_zrnikeparam.localroi_set.String);

%     sigmaxy = beads_zrnikeparam.sigmaxy;
%     STD_aberrations0 = beads_zrnikeparam.RRSE.STD_aberrations0;
%     STD_aberrations0_average = mean(STD_aberrations0);                            %每个beads的original泽尼克系数与平均泽尼克系数均方差再取平均值
value_all = value;
beads_num = length(pos);
frame_size = beads_zrnikeparam.roi;
value_std = std(transpose(cell2mat(value)));
value_mean = mean(transpose(cell2mat(value)));
ZernikeModeIndx =[]; 
frame_fov = beads_zrnikeparam.roi;
value_indx=NaN(frame_size);
if min(frame_fov)>localroi_set*2
    xy_roi = localroi_set;
else
    xy_roi = min(frame_fov);
end

for j=1:21
%     j
    for i = beads_num:-1:1        %倒叙循环是为了防止连续的beads有问题时跳过
        value_indx(pos{i}(1),pos{i}(2))=value{i}(j);     
    end
    value_pos{j} = value_indx;
    value_indx=NaN(frame_size);
    
end

for i = 1:beads_num   
%     i
      pos{i} = pos{i}(1:2);
%       pos_all{i} = pos_all{i}(1:2);
%       num_pickup{i} = pos{i};
      for j=1:21
         value_pos_crop=  value_pos{j}(max(pos{i}(1)-xy_roi/2,0)+1:min(pos{i}(1)+xy_roi/2,frame_fov(1)),max(pos{i}(2)-xy_roi/2,0)+1:min(pos{i}(2)+xy_roi/2,frame_fov(2)));
         value_pos_crop_t = reshape(value_pos_crop, size(value_pos_crop,1)*size(value_pos_crop,2),1);  %把二维数组转换成一维数组
         value_pos_crop_mean(j) = mean(value_pos_crop_t,'omitnan');   
         value_pos_crop_std(j) = std(value_pos_crop_t,'omitnan');
      end
      value_roi_mean{i}=value_pos_crop_mean;
      value_roi_std{i}=value_pos_crop_std;   
end

for j = 1:21       %前21项为像差系数map，后两项为x和y的sigma用于psf的rescale
     p.status.String=['calculate zernike coefficient maps of individual PSFs: ' num2str(j) ' of 21']; drawnow

    x = zeros(1,beads_num);
    y = zeros(1,beads_num);
    z = zeros(1,beads_num);

    for i = beads_num:-1:1
        pos{i} = pos{i}(1:2);
        if any(transpose(value_all{i})>value_roi_mean{i}+std_ref*value_roi_std{i}) || any(transpose(value_all{i})<value_roi_mean{i}-std_ref*value_roi_std{i}) || RRSE1(i)>modelrrse_set     %mark 过滤拟合不好的beads
            x(i) = [];
            y(i) = [];
            z(i) = [];
        else
             x(i) = pos{i}(1,1);
             y(i) = pos{i}(1,2);
             z(i) = value{i}(j,1);
%              if j<=21
%                  z(i) = value{i}(j,1);
%              else
%                  z(i)=sigmaxy{i}(1,j-21);  
%              end

        end
        
    end
    
    BeadsCoord=cell2mat(pos);
    X_BeadsCoord = BeadsCoord(1:2:length(BeadsCoord));
    Y_BeadsCoord = BeadsCoord(2:2:length(BeadsCoord));
    scatter(axdist,X_BeadsCoord,Y_BeadsCoord,30,'blue');
    xlim([0 frame_size(1)]);
    ylim([0 frame_size(2)]);
    hold(axdist,'on');
    scatter(axdist,x,y,30,'x','red');
    hold(axdist,'off');
    legend(axdist,'total beads','valid beads','Location','northeast');
    xlabel('X'),ylabel('Y');
    title(axdist,['total beads  N= ',num2str(beads_num), ' beads'  newline   'pick up beads N =',num2str(length(x)),'beads']);
    box(axdist,'on');
    
%     title(axdist,['total beads  N= ',num2str(beads_num),'beads'])
%     maps_show(23,frame_size(1),frame_size(2)) = [];
%     if j<=21
%        p.status.String=['calculate zernike coefficient maps of individual PSFs: ' num2str(j) ' of 21']; drawnow
       
%        [x1,y1]=meshgrid(1:1:frame_size(1),1:1:frame_size(2)); 
%        maps{j}=griddata(x,y,z,x1,y1,'v4');  
       
       F=scatteredInterpolant(x',y',z','natural');
       [x1,y1]=meshgrid(1:1:frame_size(1),1:1:frame_size(2));
       maps{j}=F(x1,y1);
       
%        sigma=40;%标准差大小
%        window=double(uint8(3*sigma)*2+1);%窗口大小一半为3*sigma  
%        H=fspecial('gaussian',window, sigma);%fspecial('gaussian',hsize, sigma)产生滤波模板 
%        maps_gauss{j}=imfilter(maps{j},H,'replicate');%为了不出现黑边，使用参数'replicate'（输入图像的外部边界通过复制内部边界的值来扩展）
       maps_gauss{j} = imgaussfilt(maps{j},mapsmooth_set);
       maps_show(:,:,j) = maps_gauss{j};
%     else
%        maps{j}=sigmaxy_avg;     
%        maps_show(:,:,j) = maps{j};
%     end
    ZernikeModeIndx_temp = {ZernikeMode(beads_zrnikeparam.model,j)};
    ZernikeModeIndx = [ZernikeModeIndx,ZernikeModeIndx_temp];
%     ZernikeMode_indx(j,1:length(ZernikeModeIndx)) = ZernikeModeIndx;
end

%     temp ={ 'probability','photons','dx','dy','znm','bg','image'};
    tags={[],[],ZernikeModeIndx,[]};
    imx(maps_show,'Parent',axmap,'Tags',tags);
%     imx(maps_show,'Parent',axmap,ZernikeMode_flage,ZernikeMode_indx)   %     under develop

    
%     imagesc(axmap,maps_gauss{2})
%     axis(axmap,'equal')
%     axis(axmap,'tight')
%     colorbar(axmap)
%     title(axmap,['Astigmatism x ' '(',num2str(beads_zrnikeparam.model(2,:)),')']); 
%     imx(maps_gauss,'Parent',axmap)


%     mapdir=strrep(outputfile,'3dcal.mat','maps_gauss.mat');
%     save(mapdir,'maps_gauss');

end