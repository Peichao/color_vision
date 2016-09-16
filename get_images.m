load('imageArray.mat')
for i = 1:5766;
    image = imageArray{i, 1};
    save(strcat('E:\color_vision\images\imageArray',num2str(i),'.mat'), 'image');
end