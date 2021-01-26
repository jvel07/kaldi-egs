#!/bin/bash

## x-vectors recipe for the dementia dataset ##

. ./path.sh || exit 1
. ./cmd.sh || exit 1

set -e
experiment_folder=exp_23mfcc_sp_noAug
mfccdir=`pwd`/${experiment_folder}/mfcc

vaddir=`pwd`/${experiment_folder}/mfcc

nnet_dir=${experiment_folder}/xvector_nnet_bea16k_sp_noAug_23mf


stage=3


# Extract MFCCs and VAD from the big bea corpus and demencia94ABC
if [ $stage -le -1 ]; then
  for name in train; do
  # Making spk2utt files
  utils/utt2spk_to_spk2utt.pl data/${name}/utt2spk > data/${name}/spk2utt
  # make mfcc features
   steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 \
     --cmd "${train_cmd}" data/${name} ${experiment_folder}/make_mfcc ${mfccdir}
   utils/fix_data_dir.sh data/${name} # fix data directory
  # compute VAD
   sid/compute_vad_decision.sh --nj 8 --cmd "${train_cmd}" \
     data/${name} ${experiment_folder}/make_vad ${vaddir}
   utils/fix_data_dir.sh data/${name} # fix data directory
  done
  echo "FINALIZED STAGE: $stage"
fi


# Section for augmenting datasets and for MFCCs computation over the augmented datasets
if [ $stage -le -2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the TRAIN list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 /media/jose/hk-data/audio/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble # 43580 utterances

  # Take a random subset of the augmentations (58000/58600)
  utils/subset_data_dir.sh data/train_aug 42000 data/train_aug_27k
  utils/fix_data_dir.sh data/train_aug_27k

  # Make MFCCs for the augmented data.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$cuda_cmd" \
    data/train_aug_27k ${experiment_folder}/make_mfcc $mfccdir

  # Combine the clean and augmented TRAIN list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_27k data/train

  # Filter out the clean + augmented portion of the TRAIN (COLD) list.  This will be used to
  # train the PLDA model later in the script.
  #utils/copy_data_dir.sh data/swbd_sre_combined data/sre_combined
  #utils/filter_scp.pl data/sre/spk2utt data/swbd_sre_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_combined/utt2spk
  #utils/fix_data_dir.sh data/sre_combined

fi


if [ $stage -le 3 ]; then
  # prepare the features to generate examples for xvector training (for running 'run_xvector_1a.sh')
  # here we apply CMVN
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 8 --cmd "$cuda_cmd" \
  data/train_combined data/train_combined_cmvn ${experiment_folder}/train_combined_cmvn
  utils/fix_data_dir.sh data/train_combined_cmvn
  echo "FINALIZED STAGE: $stage"

  # Remove features that are too short after removing silence
  # frames.  We want at least 4s (400 frames) per utterance.
#  echo 'removing short features'
#  min_len=400
#  mv data/train_combined_cmvn/utt2num_frames data/train_combined_cmvn/utt2num_frames.bak
#  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_cmvn/utt2num_frames.bak > data/train_combined_cmvn/utt2num_frames
#  utils/filter_scp.pl data/train_combined_cmvn/utt2num_frames data/train_combined_cmvn/utt2spk > data/train_combined_cmvn/utt2spk.new
#  mv data/train_combined_cmvn/utt2spk.new data/train_combined_cmvn/utt2spk
#  utils/fix_data_dir.sh data/train_combined_cmvn
#
#  # Now we're ready to create training examples.
#  utils/fix_data_dir.sh data/train_combined_cmvn
fi

# Train xvecs network
# generate configs, egs, and train model (-1 to train it from scratch)
local/nnet3/xvector/tuning/run_xvector_1a.sh --stage $stage --train-stage -1 \
 --data data/train_combined_cmvn --nnet-dir $nnet_dir --egs-dir $nnet_dir/egs
  echo "Finished training the xvecs DNN - $stage"