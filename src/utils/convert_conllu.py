import os
import sys

def read_pos_file(file_path):
    """
    Read a POS-tagged file where each line contains a word and its POS tag,
    and blank lines separate sentences.
    """
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split line into token and POS tag
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    pos = parts[1]
                    current_sentence.append((word, pos))
            else:
                # Empty line indicates end of sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
        
        # Don't forget to add the last sentence if file doesn't end with a blank line
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences

def write_conllu(sentences, output_file):
    """
    Write sentences in CoNLL-U format.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            f.write(f"# sent_id = {i+1}\n")
            f.write("# text = " + " ".join(word for word, _ in sentence) + "\n")
            
            for j, (word, pos) in enumerate(sentence):
                # CoNLL-U format:
                # ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                # We'll fill in only ID, FORM, and UPOS (POS tag)
                token_id = j + 1
                f.write(f"{token_id}\t{word}\t_\t{pos}\t_\t_\t_\t_\t_\t_\n")
            
            # Empty line between sentences
            f.write("\n")

def process_files(input_file, output_file):
    """
    Process one input file and write its CoNLL-U version.
    """
    print(f"Converting {input_file} to CoNLL-U format...")
    sentences = read_pos_file(input_file)
    write_conllu(sentences, output_file)
    print(f"Conversion complete: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_conllu.py <input_directory>")
        print("Or: python convert_to_conllu.py <input_file> <output_file>")
        sys.exit(1)
    
    # Process single file mode
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        process_files(input_file, output_file)
        return
    
    # Process directory mode
    input_dir = sys.argv[1]
    output_dir = input_dir + "_conllu"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each expected file
    for filename in ['train.txt', 'dev.txt', 'test.txt']:
        input_path = os.path.join(input_dir, filename)
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping.")
            continue
        
        # Create corresponding .conllu file
        output_filename = filename.replace('.txt', '.conllu')
        output_path = os.path.join(output_dir, output_filename)
        
        process_files(input_path, output_path)

if __name__ == "__main__":
    main()