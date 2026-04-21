% Load the data from the text files
% data_sensitive = readmatrix('Sen_MvT_pix1.txt');
% data_insensitive = readmatrix('Insen_MvT_pix1.txt');
% data_electronic = readmatrix('Elec_MvT_pix1.txt');

data_sensitive = readmatrix('Magnetic_Field_Sensitive.txt');
data_insensitive = readmatrix('Magnetic_Field_Insensitive.txt');
data_electronic = readmatrix('Magnetic_Field_Electronic.txt');

% Extract time and magnetic field data
time_sensitive = data_sensitive(:, 1); 
magnetic_field_sensitive = data_sensitive(:, 2);

time_insensitive = data_insensitive(:, 1);
magnetic_field_insensitive = data_insensitive(:, 2);

time_electronic = data_electronic(:, 1);
magnetic_field_electronic = data_electronic(:, 2);

% Sampling frequency
fs_sensitive = 1 / (time_sensitive(2) - time_sensitive(1));
L_sensitive = size(data_sensitive, 1);

fs_insensitive = 1 / (time_insensitive(2) - time_insensitive(1));
L_insensitive = size(data_insensitive, 1);

fs_electronic = 1 / (time_electronic(2) - time_electronic(1));
L_electronic = size(data_electronic, 1);

% FFT
[xdft_sensitive, psdx_sensitive, freq_sensitive] = compute_fft(magnetic_field_sensitive, fs_sensitive, L_sensitive);
[xdft_insensitive, psdx_insensitive, freq_insensitive] = compute_fft(magnetic_field_insensitive, fs_insensitive, L_insensitive);
[xdft_electronic, psdx_electronic, freq_electronic] = compute_fft(magnetic_field_electronic, fs_electronic, L_electronic);

% -----------------------------
% EXTRACT NOISE FLOOR (µT/√Hz)
% -----------------------------
valid_sensitive = (freq_sensitive > 0.1) & (freq_sensitive < 0.8*fs_sensitive/2);

sens_sensitive = median(sqrt(psdx_sensitive(valid_sensitive))) * 1e6;

% -----------------------------
% 🔥 CREATE LEGEND STRING
% -----------------------------
legend_sensitive = sprintf('Sensitive (%.4f µT/√Hz)', sens_sensitive);

% -----------------------------
% PLOT
% -----------------------------
figure;

loglog(freq_sensitive, sqrt(psdx_sensitive), '-', 'LineWidth', 1.5, ...
    'DisplayName', legend_sensitive); hold on;

loglog(freq_insensitive, sqrt(psdx_insensitive), '-',  'LineWidth', 1.5, ...
    'DisplayName', 'Insensitive');

loglog(freq_electronic, sqrt(psdx_electronic), '-',  'LineWidth', 1.5, ...
    'DisplayName', 'Electronic');

hold off;

grid on;
xlim([0.0152, 500]); 
xlabel('Frequency (Hz)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Sensitivity (T/√Hz)', 'FontSize', 16, 'FontWeight', 'bold');
title('Magnetic Field Noise Spectrum', 'FontSize', 18, 'FontWeight', 'bold');

legend('FontSize', 14, 'Location', 'best');
set(gca, 'FontSize', 14, 'LineWidth', 0.5);

% Display calculated parameters
disp(['Sampling Frequency (Sensitive): ', num2str(fs_sensitive), ' Hz']);
disp(['Number of Samples (Sensitive): ', num2str(L_sensitive)]);

disp(['Sampling Frequency (Insensitive): ', num2str(fs_insensitive), ' Hz']);
disp(['Number of Samples (Insensitive): ', num2str(L_insensitive)]);

disp(['Sampling Frequency (Electronic): ', num2str(fs_electronic), ' Hz']);
disp(['Number of Samples (Electronic): ', num2str(L_electronic)]);

disp(['Sensitivity (Sensitive): ', num2str(sens_sensitive, '%.3f'), ' µT/√Hz']);

% -----------------------------
% Helper function
% -----------------------------
function [xdft, psdx, freq] = compute_fft(magnetic_field, fs, L)
    N = L;
    xdft = fft(magnetic_field);
    xdft = xdft(1:N/2+1);
    psdx = (1 / (fs * N)) * abs(xdft).^2;
    psdx(2:end-1) = 2 * psdx(2:end-1);
    freq = 0:fs/L:fs/2;
end