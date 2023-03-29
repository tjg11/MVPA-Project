# IMPORT STATEMENTS
import scipy.io
from os import path
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
import matplotlib.pyplot as plt
import csv
print("Imported packages...")

# SUBJECT DETAILS
subject_ids = [
    'sub-NSxGxXJx1998',
    'sub-NSxGxYRx1992',
    'sub-NSxLxBNx1985',
    'sub-NSxLxIUx1994',
    'sub-NSxLxQFx1997',
    'sub-NSxLxVDx1987',
    'sub-NSxLxVJx1998',
    'sub-NSxLxYNx1999',
    'sub-NSxGxBYx1981',
    'sub-NSxGxNXx1990',
    'sub-NSxGxRFx1978'
]

# Variables for CSV labeling
subject_ids_csv = [
    'sub-NSxGxXJx1998',
    'sub-NSxGxYRx1992',
    'sub-NSxLxBNx1985',
    'sub-NSxLxIUx1994',
    'sub-NSxLxQFx1997',
    'sub-NSxLxVDx1987',
    'sub-NSxLxVJx1998',
    'sub-NSxLxYNx1999',
    'sub-NSxGxBYx1981',
    'sub-NSxGxNXx1990',
    'sub-NSxGxRFx1978'
]

header = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Fold 6"]

# Subject #10 (index 9) is skipped
subs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

# Mask names
masks = ['lMT', 'rMT','lPT', 'rPT']

# Numpy array for storing stats
cv_means = {
    'lMT': np.zeros((6, 11)),
    'rMT': np.zeros((6, 11)),
    'lPT': np.zeros((6,11)),
    'rPT': np.zeros((6,11))
}

# Set up csv file for writing out folds accuracies
f = open("sub_folds.csv", "w")
writer = csv.writer(f)
writer.writerow(header)

for which_mask in masks:
    for which_sub in subs:
    # Storage variables (need to change to Numpy matricies)

        complete_subject = []
        chunks = []
        all_conds = []
    # Get ROI 
        roi_path = path.join('brainvoyagerfiles', subject_ids[which_sub], 'rois', (subject_ids[which_sub] + which_mask + '.mat'))
        roi = scipy.io.loadmat(roi_path)
        roi = roi["mask_array"]

        for which_run in range(1, 7):

            # LOAD CONDITIONS
            cond_path = path.join('brainvoyagerfiles', subject_ids[which_sub], 'func', ('conditions' + subject_ids[which_sub] + 'AMPBrun' + str(which_run) +'.mat'))
            conds = scipy.io.loadmat(cond_path)
            conds = conds['conds']
            nevents = len(conds)

            conds_stimtype = conds[:, 1]
            print(f"Conditions: {conds_stimtype}.")
            all_conds.append(conds_stimtype)

            # Set the number of training events? (I think this is pointless with the way the function works)
            ntrainevents = 0
            ntrainevents = ntrainevents + sum(conds[:, 2] == 1)

            # LOAD GLM FILE (JUST ONE FOR NOW)
            glmpath = path.join('brainvoyagerfiles', subject_ids[which_sub], 'glm', (subject_ids[which_sub] + '-AMPBrun-' + str(which_run) + '.mat'))
            glmdata = scipy.io.loadmat(glmpath)

            print(f"{which_run}------------------Loaded GLM .mat file for: {subject_ids[which_sub]}.")

            # Reassign glmdata variable to actually be the GLMData
            glmdata = glmdata["glmbetamaps"]
            glmdata_shape = glmdata.shape

            # Segment out confounds
            glmdata = glmdata[:, :, :, 0:nevents]
            glmdata_shape = glmdata.shape


            # APPLY ROI MASK TO GLM DATA
            # This has the same functionality as cosmo_fmri_dataset (I think!!)

            # Storage of entire masked set
            complete_mask = []


            # Apply the mask to each set of voxels (need to ravel, this isn't 100% accurate)
            for pred in range(0, glmdata.shape[3]):  
                masked_glm = glmdata[:, :, :, pred].flatten()[roi.flatten() == 1]
                complete_mask.append(masked_glm)

            # Stack mask and add to subject
            complete_mask = np.stack(complete_mask, axis=0)
            complete_subject.append(complete_mask)


            # Generate chunks
            for i in range(nevents):
                chunks.append(1 * which_run)

        # Assemble entire subject using stack and reshape to correct size
        complete_subject = np.stack(complete_subject, axis = 0)
        complete_subject = complete_subject.reshape(26 *6, np.count_nonzero(roi))

        # Assemble all conditions and reshape to correct size
        all_conds = np.array(all_conds)


        # GENERATE / REASSIGN TO CHUNKS, SAMPLES, TARGETS
        print("------------------------")
        chunks = np.ravel(chunks)
        samples = complete_subject
        targets = all_conds
        targets = targets.reshape(156)
        print(f"Samples: {samples.shape}\nTargets: {targets.shape}\nChunks: {chunks.shape}")

        # TRY TO PERFORM MVPA
        clf = LinearDiscriminantAnalysis(solver='lsqr')
        cv_scores = cross_val_score(estimator=clf,
                                    X=samples,
                                    y=targets,
                                    groups=chunks,
                                    cv=LeaveOneGroupOut(),
                                    n_jobs=-1,
                                    verbose=1)

        print('Average accuracy = %.02f percent\n' % (cv_scores.mean() * 100))

        # Write data to CSV
        writer.writerow(cv_scores)

        # Save data for plotting
        idx = 0
        for score in cv_scores:
            cv_means[which_mask][idx, which_sub] = score
            idx += 1
    print(f"++++++Finished mask {which_mask}.")
    cv_means[which_mask] = np.delete(cv_means[which_mask], 9, 1)
# Try to show a figure (very much WIP)
lMT_means = []
rMT_means = []
lPT_means = []
rPT_means = []

f.close()

for subs in range(cv_means['lMT'].shape[1]):
    lMT_means.append(np.round((cv_means['lMT'][:, subs].mean()), 4) * 100)
    lMT_mean = sum(lMT_means) / len(lMT_means)

for subs in range(cv_means['rMT'].shape[1]):
    rMT_means.append(np.round((cv_means['rMT'][:, subs].mean()), 4) * 100)
    rMT_mean = sum(rMT_means) / len(rMT_means)

for subs in range(cv_means['lPT'].shape[1]):
    lPT_means.append(np.round((cv_means['lPT'][:, subs].mean()), 4) * 100)
    lPT_mean = sum(lPT_means) / (len(lPT_means)) 

for subs in range(cv_means['rPT'].shape[1]):
    rPT_means.append(np.round((cv_means['rPT'][:, subs].mean()), 4) * 100)
    rPT_mean = sum(rPT_means) / len(rPT_means)


sizes = np.random.uniform(15, 80, 10)
colors = np.random.uniform(15, 80, 10)
x_axis1 = np.linspace(0.6, 1.4, 10)
x_axis2 = np.linspace(2.6, 3.4, 10)
x_axis3 = np.linspace(4.6, 5.4, 10)
x_axis4 = np.linspace(6.6, 7.4, 10)

fig,ax = plt.subplots()

ax.scatter(x_axis1, lMT_means, color='k',alpha=0.9, zorder=1)
ax.scatter(x_axis2, rMT_means, color='k',alpha=0.9, zorder=1)
ax.scatter(x_axis3, lPT_means, color='k',alpha=0.9, zorder=1)
ax.scatter(x_axis4, rPT_means, color='k',alpha=0.9, zorder=1)

ax.grid(color='g', linestyle='-', linewidth=0.5, alpha=0.2)
ax.bar([1, 3, 5, 7], [lMT_mean, rMT_mean, lPT_mean, rPT_mean], tick_label=masks, zorder=0)

plt.xlabel('ROIS')
plt.ylabel('Accuracy')
plt.axhline(y=33, color='k', linestyle='--')
plt.ylim([0, 60])
plt.xlim([0, 9])
plt.show()




    

# 6 runs
# 4 masks
# 10 subjects
# 6 accuracies per subject, each averaged to a mean