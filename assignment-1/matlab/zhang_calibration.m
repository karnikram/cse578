clear variables
clc
close all

for i =  1:15
    %imageFileName = sprintf('img%d.jpg',i);
    %imageFileNames{i} = fullfile('../data/phone-camera/zhang',imageFileName);
    
    imageFileName = sprintf('img%d.JPG',i);
    imageFileNames{i} = fullfile('../data/zhang',imageFileName);
end

[imagePoints,boardSize] = detectCheckerboardPoints(imageFileNames);

worldPoints = generateCheckerboardPoints(boardSize, 0.029);

[params,imagesUsed,estimataionErrors] = estimateCameraParameters(imagePoints,worldPoints, ...
'EstimateSkew',true','NumRadialDistortionCoefficients',3, 'EstimateTangentialDistortion',true);

figure, imshow(imageFileNames{1}), hold on
plot(imagePoints(:,1,1),imagePoints(:,2,1),'ro-','LineWidth',2,'MarkerFaceColor','r');

plot(params.ReprojectedPoints(:,1,1),params.ReprojectedPoints(:,2,1),'bo-','LineWidth',2,'MarkerFaceColor','b');

disp('Average reprojection error:');
disp(params.MeanReprojectionError);

disp('Calibration matrix:');
disp(params.IntrinsicMatrix');

disp('Radial distortion:');
disp(params.RadialDistortion);

disp('Tangential distortion:');
disp(params.TangentialDistortion);

% [rotationMatrix, translationVector] = extrinsics(imagePoints,worldPoints,params);
% P = cameraMatrix(params,rotationMatrix,translationVector);
% 
% worldPoints = [worldPoints zeros(size(worldPoints,1),1) ones(size(worldPoints,1),1)];
% 
% projectedPoints = worldPoints * P;
% 
% projectedPoints(:,1) = projectedPoints(:,1) ./ projectedPoints(:,3);
% projectedPoints(:,2) = projectedPoints(:,2) ./ projectedPoints(:,3);
% 
% plot(projectedPoints(:,1),projectedPoints(:,2),'bo','LineWidth',2);
% plot(projectedPoints(:,1),projectedPoints(:,2),'b','LineWidth',2);
