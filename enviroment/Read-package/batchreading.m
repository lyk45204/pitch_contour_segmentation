classdef batchreading
    %read every file with the same extendion in one directory
    %   methods
    %       file2filename: get the filename of all the files in one folder
    %       filename2cell: read the data by using filenames
    
    properties
        input = 'directory';
        extensiontype = 'multiple';
        storetype = 'cell';
    end
    
    methods (Static)

        %% file2filename
        function filename = file2filename(folder,extension)
            % run every files in the same extendion in a directory.
            % input: folder(directory) 
            %        extension of the files in this folder
            % output: filename, a cell which stores the filename of every file; 
            
            list = dir(fullfile(folder, extension));           
            nFile   = length(list);
            filename = cell(nFile,1);
            for k = 1:nFile
                filename{k} = list(k).name;
            end
            filename = natsortfiles(filename);
        end
        %% filename2cell
        % read these files by using the filenames
        % input: filename: a cell storing the names of the files
        % output: output_data, a cell contains data of the file
        
        function output_data = filename2cell(input_filename, extension, table)
            n_files = length(input_filename); % the number of the files
            output_data = cell(n_files,1);
            if table == 0 %don't need import table
                switch(extension)
                    case '*.csv'
                        for i = 1:n_files
                            
                            output_data{i} = csvread(input_filename{i});
                        end
                    case '*.txt'
                        for i = 1:n_files
                            
                            output_data{i} = textread(input_filename{i});
                        end
                    case '*.wav'
                        for i = 1:n_files
                            
                            output_data{i} = audioread(input_filename{i});
                        end
                end
            else % table == 1
                for i = 1:n_files
                    T = readtable(input_filename{i});
                    output_data{i} = table2cell(T);
                end
            end

        end
        %% mainread
        function data = mainread(folder_path,subfolder,extension,table)
            % input: folder_path,subfolder, extension, table
            % output: data
            folder = strcat(folder_path,subfolder);
            filename = batchreading.file2filename(folder,extension);
            data = batchreading.filename2cell(filename, extension, table);
        end
    end
end

