#!/bin/bash
# extract_scores_robust.bash input_file output.csv

input="$1"
output="$2"

# Process the file to extract scores
awk 'BEGIN {OFS=","; header_found=0} 
    # First line with "SCORE:" is the header
    /^SCORE:/ && header_found == 0 {
        header_found = 1
        for (i=1; i<=NF; i++) {
            # For ligand-protein (score.txt):
            if ($i == "total_score") total_col = i
            if ($i == "interface_delta_B") score_col = i
            if ($i == "ligand_rms_no_super_B") rms_col = i
            
            # For protein-protein (docking.txt):
            if ($i == "I_sc") score_col = i
            if ($i == "Irms") rms_col = i
            
            if ($i == "description") desc_col = i
        }
        next
    }
    # Data lines
    /^SCORE:/ && header_found == 1 {
        print $desc_col, $total_col, $score_col, $rms_col
    }' "$input" | 
sort -t, -k2,2n > "$output"

# Add header to the output file
echo "PDB_Name,Total_Score,Interface_Score,RMSD" | cat - "$output" > temp && mv temp "$output"

echo "Top scores saved to $output (sorted by total_score)"