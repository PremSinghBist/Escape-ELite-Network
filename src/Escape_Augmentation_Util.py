from Bio import SeqIO
import csv

def convert_fasta_to_csv(input_fasta, output_csv):
    # Open the CSV file in write mode
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row #wild,mutated
        writer.writerow(['wild', 'mutated'])

        # Open the FASTA file and iterate over its records
        for record in SeqIO.parse(fasta_file, 'fasta'):
            # Extract the ID and sequence from each record
            #sequence_id = record.id
            sequence = str(record.seq)
            #Cleanse the sequence and make sure seq of lenght 20 is used 
            if len(sequence) != 20:
                continue

            # Write the ID and sequence to the CSV file
            writer.writerow([sequence, sequence])

    print('FASTA file has been converted to CSV successfully.')

if __name__ == "__main__":
    fasta_file = '/home/perm/cov/data/gen/escape_gan_seqs/non_sat_tweaked_01/fasta_8000_non_sat_tweaked_01.fasta'
    csv_file = '/home/perm/cov/data/gen/escape_gan_seqs/non_sat_tweaked_01/fasta_8000_non_sat_tweaked_01.csv'
    convert_fasta_to_csv(fasta_file, csv_file)