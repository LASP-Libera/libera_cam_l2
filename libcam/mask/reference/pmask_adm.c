/* C code for camera ADM pixel masking                        */
/* translated from adm pixel mask code developed              */
/* by Jake Gristey (jake.gristey@noaa.gov)                    */
/* by Hong Chen (hong.chen@lasp.colorado.edu)                 */
/*                                                            */
/* NOT a verbatim translation any more.  Modified 2026-10-06   */
/* so that the sample-centre rounding matches FLIGHT software  */
/* (a cast to int, which truncates) rather than the original   */
/* Python (round).  See the comment at that line.  The         */
/* ancestral, unmodified Python is kept at                     */
/* reference/pmask_adm_original.py.                            */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

int * pixel_mask_adm () {

    /* parameters */
    /*--------------------------------------------------------*/
    int i, j;
    int i0, j0;

    int percent_samp = 100;
    int max_radius_of_pix = 20;
    int vza_shift = 0;
    int vaa_shift = 0;
    int x_center_pix = 1024;
    int y_center_pix = 1024;

    double pix_spacing = 0.0055;
    double coef_ang2dist[] = {0.0, -2.07789E-6, -1.16828E-5, 9.92181E-2, 0.0};
    int n_coef;

    double re = 6371.0;
    double alt = 824.0;
    double pi = 3.14159;

    int vza_boundaries[] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    int vaa_boundaries[] = {0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360};

    int n_vza_bins = 9;
    int n_vaa_bins = 18;
    int ** bin_nums_all;

    int n_bins;
    int n_samples;
    int * selected_bin_nums;

    int vza_i;
    int vaa_i;
    int bin_num;

    double vza;
    double vaa;

    double vza_sat_ref;

    double dist_from_center;
    double x_dist_from_center;
    double y_dist_from_center;

    int * x_pix_selected;
    int * y_pix_selected;

    int ** mask;
    int xp = 2048;
    int yp = 2048;
    int x_pix_box_s;
    int x_pix_box_e;
    int y_pix_box_s;
    int y_pix_box_e;
    double radius_from_central_pix;

    FILE *f1;
    /*--------------------------------------------------------*/


    /* initialize <bin_nums_all> */
    /*--------------------------------------------------------*/
    bin_nums_all = (int**)malloc(sizeof(int*)*n_vza_bins);

    for (i=0; i<n_vza_bins; i++) {
        bin_nums_all[i] = (int*)malloc(sizeof(int)*n_vaa_bins);
    }

    for (i=0; i<n_vza_bins; i++) {
        for (j=0; j<n_vaa_bins; j++) {
            bin_nums_all[i][j] = j + i*n_vaa_bins;
        }
    }
    /*--------------------------------------------------------*/


    /* get <selected_bin_nums> */
    /*--------------------------------------------------------*/
    n_bins = n_vza_bins * n_vaa_bins;
    n_samples = floor(percent_samp / 100.0 * n_bins);
    selected_bin_nums = (int*)malloc(sizeof(int)*n_samples);
    for (i=0; i<n_samples; i++) {
        selected_bin_nums[i] = floor(n_bins / n_samples * i);
    }
    /*--------------------------------------------------------*/


    /* get <x_pix_selected> and <y_pix_selected> */
    /*--------------------------------------------------------*/
    n_coef = sizeof(coef_ang2dist)/sizeof(coef_ang2dist[0]);
    x_pix_selected = (int*)malloc(sizeof(int)*n_samples);
    y_pix_selected = (int*)malloc(sizeof(int)*n_samples);

    for (i=0; i<n_samples; i++) {

        bin_num = selected_bin_nums[i];
        vza_i = bin_num / n_vaa_bins;
        vaa_i = bin_num % n_vaa_bins;
        vza = vza_boundaries[vza_i] + 0.5*(vza_boundaries[vza_i+1]-vza_boundaries[vza_i]) + vza_shift;
        vaa = vaa_boundaries[vaa_i] + 0.5*(vaa_boundaries[vaa_i+1]-vaa_boundaries[vaa_i]) + vaa_shift;

        /* to use <floor>, we will need `-lm` flag in compiling */
        vza = vza - floor(vza/90.0);
        vaa = vaa - floor(vaa/360.0);

        vza_sat_ref = asin(sin(vza/180.0*pi) * (re/(re+alt))) / pi * 180.0;

        dist_from_center = 0.0;
        for (i0=0; i0<n_coef; i0++) {
            dist_from_center = dist_from_center + coef_ang2dist[i0]*pow(vza_sat_ref, n_coef-i0-1);
        }

        x_dist_from_center = dist_from_center * sin(vaa/180.0*pi);
        y_dist_from_center = dist_from_center * cos(vaa/180.0*pi);

        /* TRUNCATE, do not round.  Changed 2026-10-06.
         *
         * This file began as a faithful C translation of the original Python,
         * which used round().  Flight software does not: WFOVPixelMask_ADM.cpp
         * :: getSelectedADMBins() writes
         *
         *     (int)(xcenter + x_dist_from_center / WFOV_MASK_ADM_PIX_SPACING)
         *
         * and a C cast to int truncates toward zero.  Since these coordinates
         * are around +1000 it always floors, putting every sample centre up to
         * one pixel closer to the origin than rounding would.
         *
         * The header of this file calls it "the version that matches flight
         * software structure".  Until today that was not true of this line, and
         * nobody noticed because the Python matched the C and both differed
         * from flight in the same direction.  libcam/mask/adm.py was corrected
         * first; this follows it, so that the reference the Python is checked
         * against is the flown behaviour and not the ancestral one.
         */
        x_pix_selected[i] = (int)(x_center_pix + x_dist_from_center/pix_spacing);
        y_pix_selected[i] = (int)(y_center_pix + y_dist_from_center/pix_spacing);
    }
    /*--------------------------------------------------------*/


    /* initialize <mask> (zero filled) */
    /*--------------------------------------------------------*/
    mask = (int**)malloc(sizeof(int*)*xp);
    for (i=0; i<xp; i++) {
        mask[i] = (int*)malloc(sizeof(int)*yp);
    }

    for (j=0; j<yp; j++) {
        for (i=0; i<xp; i++) {
            mask[i][j] = 0;
        }
    }
    /*--------------------------------------------------------*/


    /* get <mask>  */
    /*--------------------------------------------------------*/
    for (i=0; i<n_samples; i++) {

        x_pix_box_s = floor(x_pix_selected[i] - max_radius_of_pix);
        x_pix_box_e = floor(x_pix_selected[i] + max_radius_of_pix + 1);
        y_pix_box_s = floor(y_pix_selected[i] - max_radius_of_pix);
        y_pix_box_e = floor(y_pix_selected[i] + max_radius_of_pix + 1);
        if (x_pix_box_s<0) {
            x_pix_box_s = 0;
        }
        if (x_pix_box_e>xp) {
            x_pix_box_e = xp;
        }
        if (y_pix_box_s<0) {
            y_pix_box_s = 0;
        }
        if (y_pix_box_e>yp) {
            y_pix_box_e = yp;
        }

        for (i0=x_pix_box_s; i0<x_pix_box_e; i0++) {
            for (j0=y_pix_box_s; j0<y_pix_box_e; j0++) {

                radius_from_central_pix = pow((pow(i0-x_pix_selected[i], 2.0)+ pow(j0-y_pix_selected[i], 2.0)), 0.5);

                if (radius_from_central_pix<=max_radius_of_pix+0.5) {
                    if ((i0<x_center_pix*2) && (j0<y_center_pix*2)) {
                        mask[i0][j0] = 1;
                    }
                }
            }
        }
    }
    /*--------------------------------------------------------*/


    /*output data for testing*/
    /*--------------------------------------------------------*/
    f1 = fopen("test-c.txt", "w");
    for (i=0; i<xp; i++) {
        for (j=0; j<yp; j++) {
            if (j<yp-1) {
                fprintf(f1, "%5d,", (int)mask[i][j]);
            }
            else {
                fprintf(f1, "%5d", (int)mask[i][j]);
            }
        }
        fprintf(f1, "\n");
    }
    fclose(f1);
    /*--------------------------------------------------------*/


    /*deallocate large sized arrays*/
    /*--------------------------------------------------------*/
    free(selected_bin_nums);
    free(x_pix_selected);
    free(y_pix_selected);

    for (i=0; i<n_vza_bins; i++) {
        free(bin_nums_all[i]);
    }
    free(bin_nums_all);

    for (i=0; i<xp; i++) {
        free(mask[i]);
    }
    free(mask);
    /*--------------------------------------------------------*/
    /*!!!!!!!!!!!!!!!!checked until here!!!!!!!!!!!!!!!!!!!!!!*/

    return 0;
}

int main () {

    int *p;

    p = pixel_mask_adm();

    return 0;
}
