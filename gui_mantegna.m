clc;
clear all;
warning off;
chos=0;
possibility=6;

while chos~=possibility,
    
    chos=menu('DIGITAL WATERMARKING','Select Video','Convert the video into frames','Show Watermarked Image','Choose Attacks','Convert Frames into Video','EXIT');
    
    if chos==1
        [filename pathname]= uigetfile('*.avi','select any video');
        workingDir = pathname;
        coverVideo = VideoReader(filename);
    end
    
    if chos==2
        ii = 1;
        while hasFrame(coverVideo)
            img = readFrame(coverVideo);
            img=imresize(img,[256 256]);
            filename = [sprintf('%03d',ii) '.png'];
            fullname = fullfile(workingDir,filename);
            imwrite(img,fullname) % Write out to a JPEG file (img1.jpg, img2.jpg, etc.)
            ii = ii+1;
        end
    end
    if chos==3
        
        embed_sum = 0;
        terms = 2;
		loop_var=2;
        embed(1)=1;
        s = '00000001';
        imageNames = dir(fullfile(workingDir,'*.png'));
        imageNames = {imageNames.name}';
        ii=1;
        while(ii<80)
            fid=imread([sprintf('%03d',ii) '.png']);
            fid1=fid;
            fidy=fid1(:,:,1);
            data=imresize(fidy,[256 256]);
            data1=data;
            blocksize=8;
            i = 0;
            DCT_trans=zeros(8);
            for j = 0: blocksize - 1
                DCT_trans(i + 1, j + 1) = sqrt(1 / blocksize) ...
                    * cos ((2 * j + 1) * i * pi / (2 * blocksize));
            end
            for i = 1: blocksize - 1
                for j = 0: blocksize - 1
                    DCT_trans(i + 1, j + 1) = sqrt(2 / blocksize) ...
                        * cos ((2 * j + 1) * i * pi / (2 * blocksize));
                end
            end
            sz = size(data);
            rows = sz(1,1); % finds image's rows and columns
            cols = sz(1,2);
            p=1;
            for row = 1: blocksize: rows
                for col = 1: blocksize: cols
                    DCT_matrix = data(row: row + blocksize-1, col: col + blocksize-1);
                    DCT_matrix = DCT_trans *double( DCT_matrix )* DCT_trans';
                    jpeg_img(row: row + blocksize-1, col: col + blocksize-1) = DCT_matrix;
                    sum1(p)=DCT_matrix(1,1);
                    p=p+1;
                end
            end
            for jj=ii+1:80
                fid11=imread([sprintf('%03d',jj) '.png']);
                fid12=fid11;
                fidy11=fid11(:,:,1);
                data11=imresize(fidy11,[256 256]);
                data12=data11;
                blocksize11=8;
                i1 = 0;
                DCT_trans11=zeros(8);
                for j1 = 0: blocksize11 - 1
                    DCT_trans11(i1 + 1, j1 + 1) = sqrt(1 / blocksize11) ...
                        * cos ((2 * j1 + 1) * i1 * pi / (2 * blocksize11));
                end
                for i1 = 1: blocksize11 - 1
                    for j1 = 0: blocksize11 - 1
                        DCT_trans11(i1 + 1, j1 + 1) = sqrt(2 / blocksize11) ...
                            * cos ((2 * j1 + 1) * i1 * pi / (2 * blocksize11));
                    end
                end
                sz1 = size(data11);
                rows11= sz1(1,1); % finds image's rows and columns
                cols11 = sz1(1,2);
                q=1;
                for row11 = 1: blocksize11: rows11
                    for col11 = 1: blocksize11: cols11
                        DCT_matrix11 = data11(row11: row11 + blocksize11-1, col11: col11 + blocksize11-1);
                        DCT_matrix12=DCT_matrix11-DCT_matrix11;
                        DCT_matrix11 = DCT_trans11 *double( DCT_matrix11 )* DCT_trans11';
                        jpeg_img1(row11: row11 + blocksize11-1, col11: col11 + blocksize11-1) = DCT_matrix11;
                        sum2(q)=DCT_matrix11(1,1);
                        q=q+1;
                    end
                end
                var_sum=0;
                for x=1:1024
                    sum=sum2(x)-sum1(x);
                    var_sum=var_sum + sum;
                end
                variance(jj) = var_sum*var_sum / 1024;
                variance(jj);
                if(jj>2)
                    minimum=min(variance(jj),variance(jj-1));
                    alpha(jj)=(variance(jj)-variance(jj-1))/minimum;
                    
                end
                %%%%%%%%%%%%%%%%%%% condition for scene change
                if(jj>2)
                    if(alpha(jj)>2 & variance(jj)>300)
                        filename = [sprintf('%03d',jj-1) '.png'];
                        fullname = fullfile(workingDir,filename);
                        %embed(terms)=jj-1;
                        %jj-1
                        terms = terms+1;
                        str = dec2bin(jj-1,8);
                        s = strcat(s,str);
                        %str
                        imageinput = imread(fullname);
                        [r c p]=size(imageinput);
                        if p==3
                            imageinput=rgb2gray(imageinput);
                        end
                        P1=im2double(imageinput);
                        P=imresize(P1,[256 256]);
                        [LL,LH,HL,HH] = dwt2(P,'haar','d');
                        imw2=imread('watermark.jpg');
                        [r c p]=size(imw2);
                        if p==3
                            imw2=rgb2gray(imw2);
                        end
                        watermark=im2double(imw2);
                        watermark=imresize(watermark,[128 128]);
                        min1 = cuckoo_search_mantegna(25);
                        %min1 = 8;
                        Watermarkedimage1=LL+min1*watermark;
                        min2 = cuckoo_search_mantegna(25);
                        %min2 = 2;
                        Watermarkedimage2=HH+min2*watermark;		
						Watermarkedimage_final=idwt2(Watermarkedimage1,LH,HL,Watermarkedimage2,'haar');
						embed_min1(jj-1) = min1;   %saving corresponding min1 in the array
						embed_min2(jj-1) = min2;   %saving corresponding min2 in the array
            
						for i = 1:128
							for k = 1:128
								embed_LL(jj-1,i,k) = LL(i,k);  %saving corresponding LL in the array
								embed_HH(jj-1,i,k) = HH(i,k);  %saving corresponding HH in the array
							end;
						end
            
						for i = 1:256
							for k = 1:256
								embed_wm(jj-1,i,k) = Watermarkedimage_final(i,k);  %saving corresponding pixel position in the array
							end;
                        end
                        
                        embed_sum = embed_sum + Watermarkedimage_final;
                        imwrite(Watermarkedimage_final, fullfile(fullname), 'png');
						
						%loop_var = loop_var + 1;
						
                        break
                    end
                end
            end
            ii=jj;
            alpha(1);
            alpha(2);
            alpha(3);
            %terms
            out = dec2hex(bin2dec(num2str(reshape(s,4,[])','%1d')))';
            
            
        end
        imageinput1 = imread('001.png');
        [r c p]=size(imageinput1);
        if p==3
            imageinput1=rgb2gray(imageinput1);
        end
        P11=im2double(imageinput1);
        P2=imresize(P11,[256 256]);
        [LL,LH,HL,HH] = dwt2(P2,'haar','d');
        imw22=imread('watermark.jpg');
        [r c p]=size(imw22);
        if p==3
            imw22=rgb2gray(imw22);
        end
        watermark1=im2double(imw22);
        watermark1=imresize(watermark1,[128 128]);
        min1 = cuckoo_search_mantegna(25);
        %min1 = 8;
        Watermarkedimage11=LL+min1*watermark1;
        min2 = cuckoo_search_mantegna(25);
        %min2 = 2;
        Watermarkedimage22=HH+min2*watermark1;
		
		Watermarkedimage_final1=idwt2(Watermarkedimage11,LH,HL,Watermarkedimage22,'haar');
		
		embed_min1(1) = min1;   %saving corresponding min1 in the array
		embed_min2(1) = min2;   %saving corresponding min2 in the array
            
		for i = 1:128
            for k = 1:128
				embed_LL(1,i,k) = LL(i,k);  %saving corresponding LL in the array
				embed_HH(1,i,k) = HH(i,k);  %saving corresponding HH in the array
			end;
		end
            
		for i = 1:256
            for k = 1:256
				embed_wm(1,i,k) = Watermarkedimage_final1(i,k);  %saving corresponding pixel position in the array
			end;
		end		
		embed_sum = embed_sum + Watermarkedimage_final1;
        imwrite(Watermarkedimage_final1, '001.png', 'png');
        Watermark_image = embed_sum/terms;
        pic1 = P;   
        pic2 = Watermark_image;
        psnr = PSNR(pic1,pic2)
        out
    end
	

    if chos==4
			
                
        chos_attack=menu('DIFFERENT ATTACKS','No Attack','Gamma Correction','Median Filtering','Crop','Rotation','Histogram Equilisation','Gaussian Noise','Motion Blur','Resizing','Contrast Adjustment','Image Sharpening');
        switch(chos_attack)
            
            case 1
                
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                        for k = 1:128
                            LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                            HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions    
                        end;
                    end
                    
					attackedImage = frame_image;
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2) 
                
            case 2
                
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                         for k = 1:128
                             LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                             HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                         end;
                    end
                     
                    attackedImage = gammacorrection(frame_image,1.5);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2) 
            
            case 3
                
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                attacked_frame=0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                         for k = 1:128
                             LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                             HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                         end;
                    end
                    
					attackedImage = medianAttack(frame_image);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2)
                
            case 4
                
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                attacked_frame=0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                        for k = 1:128
                             LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                             HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                        end;
                    end
                    
					attackedImage = cropAttack(frame_image);
                    attackedImage = imresize(attackedImage, [256 256]);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2) 
                
            case 5
                
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                attacked_frame=0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                        for k = 1:128
                             LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                             HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                        end;
                    end

					attackedImage = rotationAttack(frame_image, -20);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2) 
                 
                
            case 6
               
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                attacked_frame=0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                        for k = 1:128
                            LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                            HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                        end;
                    end

					attackedImage = histattack(frame_image);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2)  
                
                
            case 7
                
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                attacked_frame=0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                        for k = 1:128
                            LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                            HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                        end;
                    end

					attackedImage = gaussianNoise(frame_image);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2)  
                
            case 8
                
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                attacked_frame=0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                         for k = 1:128
                            LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                            HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                         end;
                    end
                    
					attackedImage = motionattack(frame_image);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2)  
                
                
            case 9
                
                
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                attacked_frame=0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                        for k = 1:128
                            LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                            HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                        end;
                    end

					attackedImage = resizeAttack(frame_image);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2) 
                
                
            case 10
                
                 
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                attacked_frame=0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                        for k = 1:128
                            LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                            HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                        end;
                    end

					attackedImage = contrastattack(frame_image);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2) 
                
                 
            case 11
                
               
                attacked_frame=0;
                sum_LL = 0;
                sum_HH = 0;
                sum_min1 = 0;
                sum_min2 = 0;
                attacked_frame=0;
                r_image = 0;
                imageNames = dir(fullfile(workingDir,'*.png'));
                imageNames = {imageNames.name}';
                x = inputdlg('ENTER THE SECURITY KEY:',...
                    'Sample', [1 100]);
                data = char(x);
                j=1;
                while (j<length(data))
                    d = strcat(data(j),data(j+1));
                    d1 = hex2dec(d);
                    for i = 1:256
                        for k = 1:256
                            frame_image(i,k) = embed_wm(d1,i,k);   %extracting the corresponding pixel positions
                        end;
                    end
                     
                    for i = 1:128
                        for k = 1:128
                            LL1(i,k) = embed_LL(d1,i,k); %extracting the corresponding LL positions
                            HH1(i,k) = embed_HH(d1,i,k); %extracting the corresponding HH positions
                        end;
                    end

					attackedImage = sharpattack(frame_image);
					attacked_frame=attacked_frame + attackedImage;
                    sum_LL = sum_LL + LL1;
                    sum_HH = sum_HH + HH1;
                    sum_min1 = sum_min1 + embed_min1(d1);
                    sum_min2 = sum_min2 + embed_min2(d1);
                    j=j+2;
                end   

                attacked_frame = attacked_frame/terms;
                sum_LL = sum_LL/terms;
                sum_HH = sum_HH/terms;
                sum_min1 = sum_min1/terms;
                sum_min2 = sum_min2/terms;
                imshow(attacked_frame);
                
                [a b c d]=dwt2(attacked_frame,'haar','d');
                recovered_image1=((a-sum_LL)/sum_min1);
                recovered_image2=((d-sum_HH)/sum_min2);
                recovered_image = (recovered_image1+recovered_image2)/2;
                imshow(recovered_image);
                
                pic1 = watermark;
                pic2 = recovered_image;
                nc = NC(pic1,pic2) 
                
        end
    
    end
    
    if chos==5
        
        
        imageNames = dir(fullfile(workingDir,'*.png'));   
        imageNames = {imageNames.name}';
        outputVideo = VideoWriter(fullfile(workingDir,'output.avi'));
        outputVideo.FrameRate = coverVideo.FrameRate;
        open(outputVideo)
        for ii = 1:length(imageNames)
            img = imread(fullfile(workingDir,imageNames{ii}));
            writeVideo(outputVideo,img)
        end
        close(outputVideo)
    end
    
end    
