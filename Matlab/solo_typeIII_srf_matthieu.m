tds_mode = 'xld1';%convertCharsToStrings(data.tds_config_label(:, index));
% Pachenko's antenna angle
pacang = 125.;%132.2;
% RPW ANT in SRF 
V1=[0, 1];
% Geometric antenna
%V2=[0.81915204586197221, -0.57357643410459713]; 
%V3=[-0.81915204586197221, -0.57357643410459713]; 
% Pachenko antenna
V2=[sind(pacang), cosd(pacang)]; 
V3=[-sind(pacang), cosd(pacang)]; 

% XLD1 TDS mode
V1V3 = V1 - V3; %CH1
V2V1 = V2 - V1; %CH2
if contains(tds_mode, 'SE1')
% SE1 TDS mode
    M(1,:) = V1; %CH1
    M(2,:) = V2; %CH2
    tds_mode = 'se1';
else 
    % construct transformation matrix ANT->SRF 
    M(1,:) = V1V3; % V1-V3
    M(2,:) = V2V1; % V2-V1
    % DIFF1 TDS mode is same
    if contains(tds_mode, 'DIFF1')
        tds_mode = 'diff1';
    else
        tds_mode = 'xld1';
    end
end
M = inv(M);
t0 = datenum(2021, 10, 09, 6, 30, 0);
t1 = datenum(2021, 10, 09, 9, 0, 0);

idd = find(data.epoch >= t0 & data.epoch < t1);

ep = data.epoch(idd);
smat = zeros(256, 2, 2, length(idd));
for i=1:length(idd)
    index = idd(i);
    time = ep(i);
% load TDS TSWF data
% if it's at the OKF then use find_file
    nsamp = double(data.samples_per_ch(index));
    ww = data.data(:,1:nsamp,index);
% projection: E = MAT(ANT->SRF) * V; where MAT(2,2) and V is observed field 
E = M*ww(1:2,:) ; % transformation into SRF (Y-Z)
E0 = ww(1:2,:) ;
dt = 1./data.samp_rate(index);
maxfq = 150e3;
fftlen = 512;avg=31;ttags=[];tstep=[];ovrlp=1;sm=1;plotit=0;
%avg = 5; % SM averadging
[sp, tt, fq, ssm, nav] = make_spectrogram(E0, fftlen, dt, maxfq, avg, ttags, tstep, ovrlp, sm, plotit);
% smooth Spectral matrix
%sm = avg_sm(ssm, 5, 5); % smooth SM with 5x5 window in time and frequency
smat(:,:,:,i) = squeeze(ssm);
end
idx = find(fq < 205e3);fq=fq(idx);
smat=smat(idx, :, :, :); 
set(gcf, 'PaperType', 'a4', 'PaperOrientation', 'landscape');
figure(1)
%sp(:,:,1) = sm(:,:,1,1);
%sp(:,:,2) = sm(:,:,2,2);
% Spectram matrices
Sxx = squeeze(smat(:,1,1,:)); Syy = squeeze(smat(:,2,2,:)); Sxy = squeeze(smat(:,1,2,:));
% ESUM, i.e. Stokes' S
ESUM = Sxx + Syy;
subplot(4,1,1)
opts.colorax = [-14. -9.];
opts.linear_color = 0;
opts.cbar_label = 'ESUM';
plot_spectrogram(ESUM, fq, ep,opts);
title('RPW/TDS SURV-RSWF');
Cyz = Sxy ./ sqrt( Sxx .* Syy ); 
% coherence
Cm = abs(Cyz);
opts.colorax = [0. 1.];
opts.linear_color = 1;
opts.cbar_label = 'Coherence';
subplot(4,1,2)
plot_spectrogram(Cm, fq, ep,opts);
% phase delay
Cp = rad2deg(angle(Cyz));
opts.colorax = [-180. 180.];
opts.linear_color = 1;
opts.cbar_label = 'Phase delay';
subplot(4,1,3)
plot_spectrogram(Cp, fq, ep,opts);

subplot(4,1,4)
Erat = Sxx ./ Syy;
opts.colorax = [.6 1.2];
opts.linear_color = 1;
opts.cbar_label = 'PSD_{EY} / PSD_{EZ}';
plot_spectrogram(Erat, fq, ep,opts);
figure(2)
clf;
ii=find(fq > 150e3);
semilogx(fq(ii), squeeze(Sxx(ii,1)))
title(strcat('TDS-RSWF', datestr(ep(1), 31)));
xlabel('Frequency [Hz]');
ylabel('PSD')

% set threshold for coherence and power > 80%
saveas(gcf,'solo_tds_rswf_typeIII_srf_v12.png')
