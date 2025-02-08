class DataLoader:
    @staticmethod
    def load_txt_data(file_path):
        """
        Load data from txt file with format: label id1 id2 text1 text2
        Returns only labels and text pairs
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
            # Skip header
            data = []
            for line in lines[1:]:  # Skip header row
                parts = line.strip().split('\t')
                if len(parts) >= 5:  # Ensure line has all required parts
                    label = int(parts[0])
                    text1 = parts[3]
                    text2 = parts[4]
                    data.append((label, text1, text2))
            
            # Separate into labels and text pairs
            labels = [item[0] for item in data]
            text_pairs = [(item[1], item[2]) for item in data]
            
            return text_pairs, labels
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None

# import csv

# class DataLoader:
#     @staticmethod
#     def load_txt_data(file_path, max_rows=5000):
#         """
#         Load data from CSV file with columns: id, text1id, text2id, text1, text2, label
#         Returns only labels and text pairs, limited to max_rows (default: 5000)
#         """
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 reader = csv.reader(file)
#                 next(reader)  # Skip header row
                
#                 data = []
#                 for i, row in enumerate(reader):
#                     if i >= max_rows:  # Stop after max_rows
#                         break
                    
#                     if len(row) >= 6:  # Ensure row has all required parts
#                         label = int(row[5])  # Label is in the 6th column (index 5)
#                         text1 = row[3]  # text1 is in the 4th column (index 3)
#                         text2 = row[4]  # text2 is in the 5th column (index 4)
#                         data.append((label, text1, text2))
                    
            
#             # Separate into labels and text pairs
#             labels = [item[0] for item in data]
#             text_pairs = [(item[1], item[2]) for item in data]
            
#             return text_pairs, labels
#         except Exception as e:
#             print(f"Error loading data: {str(e)}")
#             return None, None
