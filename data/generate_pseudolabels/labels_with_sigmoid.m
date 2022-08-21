for split = 1:1
    load([num2str(split) '_label_livec+koniq+spaq+flive.mat']);
    data = LSRQ;

    imagename = data(:,1);
    label_livec = data(:,2);
    label_koniq = data(:,3);
    label_spaq = data(:,4);
    label_flive = data(:,5);

    label_livec_double = zeros(1,100000,'single');
    label_koniq_double = zeros(1,100000,'single');
    label_spaq_double = zeros(1,100000,'single');
    label_flive_double = zeros(1,100000,'single');

    for i = 1:100000
        label_livec_double(1,i) = single(str2double(label_livec(i)));
        label_koniq_double(1,i) = single(str2double(label_koniq(i)));
        label_spaq_double(1,i) = single(str2double(label_spaq(i)));
        label_flive_double(1,i) = single(str2double(label_flive(i)));
    end
    
    sel = randperm(100000);
    train_sel = sel(1:round(100000));
    train_path = imagename(train_sel);
    label_livec_sel = label_livec_double(train_sel);
    label_koniq_sel = label_koniq_double(train_sel);
    label_spaq_sel = label_spaq_double(train_sel);
    label_flive_sel = label_flive_double(train_sel);

    %for train split  
    fid = fopen(fullfile('./LSRQ_label/',[num2str(split) '_train_livec+koniq+spaq+flive.txt']),'w');
    
    for i = 1:500000
        rng('shuffle');
        path1_index = randi(100000,1);
        path2_index = randi(100000,1);
        path1 =  fullfile(train_path(path1_index));
        path1 = strrep(path1,'\','/');

        path1_label_livec = label_livec_sel(path1_index);
        path1_label_koniq = label_koniq_sel(path1_index);
        path1_label_spaq = label_spaq_sel(path1_index);
        path1_label_flive = label_flive_sel(path1_index);

        path2 = fullfile(train_path(path2_index));
        path2 = strrep(path2,'\','/');
        path2_label_livec = label_livec_sel(path2_index);
        path2_label_koniq = label_koniq_sel(path2_index);
        path2_label_spaq = label_spaq_sel(path2_index);
        path2_label_flive = label_flive_sel(path2_index);
        y1 =   1 ./ (1 + exp(-(path1_label_livec-path2_label_livec)));
        y2 =   1 ./ (1 + exp(-(path1_label_koniq-path2_label_koniq)));
        y3 =   1 ./ (1 + exp(-(path1_label_spaq-path2_label_spaq)));
        y4 =   1 ./ (1 + exp(-(path1_label_flive-path2_label_flive)));
        yb = (y1 + y2 + y3 + y4) / 4;
          fprintf(fid,'%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\r',path1{1},path2{1}, yb,path1_label_livec,path2_label_livec,path1_label_koniq,path2_label_koniq,path1_label_spaq,path2_label_spaq,path1_label_flive,path2_label_flive,y1,y2,y3,y4);
        if mod(i,1000)== 0
            disp(i);
        end
    end
end
    fclose(fid);
disp('completed!');