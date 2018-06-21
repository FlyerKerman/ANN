function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');%也就是说魔数的下一个4字节是图像的总数？
%再往后8个字节应该是表示了每个图像的大小
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');%行数
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');%列数

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);%reshape按照列来填充，即逐列填充
images = permute(images,[2 1 3]);%permute重新排列维度，[2 1 3]表示新的维度顺序。
%注意permute不是优先按列填充，而是按照新的维度顺序直接翻转
%上面的程序先reshape，得到优先按列填充的矩阵，之后用permute将前两个维度对调，得到相当于优先按行填充的矩阵

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));%变成每个图像对应一个列向量？
% Convert to double and rescale to [0,1]
images = double(images) / 255;
%images的第三维度表示图片的数量，所以最终应该是一个60000个图像，每个图像有784个像素
end