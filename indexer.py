import json
from bs4 import BeautifulSoup
import nltk
import os
import re
from collections import defaultdict
from nltk.stem import PorterStemmer
import heapq


LOG_FILE = "log.txt"
porter_stemmer = PorterStemmer()
total_documents = 0
total_unique_words = 0
merged_index_size = 0
total_partial_indexes = 0
urls = dict()

# File to monitor the indexer and output report questions
def log_write(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        if type(message) == str:
            log_file.write(message + "\n")




# Checks if the json file is parseable and doesn't give any UnicodeDecode errors
def is_valid_file(file_path):
    try:
        with open(file_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
            return True
    except UnicodeDecodeError:
        return False




# Return a list of all the full json file paths
def get_json_files():
    file_paths = []

    for root, sub, files in os.walk("DEV"):
        for f in files:
            f_path = os.path.join(root, f)
            if is_valid_file(f_path): # Add filepath to list only if valid
                file_paths.append(f_path)
    
    # Update the total number of documents retrieved
    global total_documents
    total_documents = len(file_paths)
    return file_paths




def get_file_content(json_file_path):
    # Open file and load the data
    with open(json_file_path, 'r', encoding = 'utf-8') as f:
        data = json.load(f)

    url = data.get("url", "Not Found")
    
    # Check if there is any content for this url
    content = data.get("content")
    if not content:
        return []

    # Parse the text in the content
    soup = BeautifulSoup(content, 'html.parser')
    weighted_text = []
    duplicate_check = set() # Keep track of the text gotten from the important tags to avoid dupliation in the regular text

    # Goes through these tags for imporant text
    for tag in ["b", "strong", "h1", "h2", "h3", "title"]:
        important_places = soup.find_all(tag)
        for place in important_places:
            text = place.get_text().strip()
            if text:
                weighted_text.append((text, tag))
                duplicate_check.add(id(place))
    
    # Finds the regular text. Text that is already a part of the important text is skipped
    regular_places = soup.find_all(string=True)
    for place in regular_places:
        if id(place.parent) not in duplicate_check:
            text = str(place).strip()
            if text:
                weighted_text.append((text, "body"))

    return weighted_text, url
        
        


# Returns a list of tuples that contain the token and its field type. ("the", "title") ("ICS", "body")
def process_text(weighted_text):
    processed_text = []

    for text, field in weighted_text:
        # re.findall here gets all alphanumeric tokens in the content of the url. O(n) time complexity as it goes through the string.
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower()) 
        for token in tokens:
            stemmed = porter_stemmer.stem(token) # Stem the token
            processed_text.append((stemmed, field))

    return processed_text
    




# Writes the partial index to a file on disk
def to_disk(inverted_index, partial_disks_count) -> None:
    name = f"INDEX/partial_index_{partial_disks_count}.json"
    with open(name, 'w', encoding='utf-8') as f:
        for token, postings in sorted(inverted_index.items()):
            for doc_id in postings:
                postings[doc_id]["fields"] = list(postings[doc_id]["fields"])
            # Put each token on its own line for visibility and use of readline() function later on
            f.write(json.dumps({token: postings}) + "\n")




def create_partial_index() -> None:
    inverted_index = defaultdict(lambda: defaultdict(lambda: {"fields": set(), "freq":0}))
    doc_id = 0
    TOKEN_LIMIT = 30000
    partial_disks_count = 0
    global urls

    # Get all the json files
    json_file_paths = get_json_files()

    for file_path in json_file_paths:
        # Process the text in the file
        weighted_text, url = get_file_content(file_path)
        urls[doc_id] = url
        tokenized_text = process_text(weighted_text)

        # Update the frequency of the token in the specific document based on weight
        for token, field in tokenized_text:
            if field != "body":
                inverted_index[token][doc_id]["fields"].add(field)
            inverted_index[token][doc_id]["freq"] += 1
        doc_id += 1  # Increment doc id to give the next doc a new number

        # Move data from main memory to disk if token limit is reached
        if len(inverted_index) > TOKEN_LIMIT:
            to_disk(inverted_index, partial_disks_count)
            inverted_index.clear()
            partial_disks_count += 1
    
    # In the end, if there were anymore tokens left in main memory before reaching the limit, move those to disk too
    if inverted_index:
        partial_disks_count += 1
        to_disk(inverted_index, partial_disks_count)

    # Update the count of the partial indexes created
    global total_partial_indexes
    total_partial_indexes = partial_disks_count





def merge_indexes():

    merged_index = defaultdict(lambda: defaultdict(lambda: {"fields": set(), "freq":0}))
    output_file = open("INDEX/merged_index.json", 'w', encoding='utf-8')
    partial_index_paths = sorted(os.listdir("INDEX"))
    min_heap = []
    opened_files = dict()
    buffers = dict()
    unique_tokens = set()
    offsets = dict()


    for fi in partial_index_paths:
        # Open every partial index and store the pointer in a dictionary
        opened_files[fi] = open(f"INDEX/{fi}", 'r', encoding='utf-8')

        # Set up the buffer for the partial index in its own dictionary
        buffers[fi] = []

        # Loop until 500 tokens was reached for each partial index
        for i in range(500):
            # Get a line from the partial index. Each line is one token.
            json_line = opened_files[fi].readline().strip()
            if not json_line:
                break
            index = json.loads(json_line)
            if index:
                # Get the item in the dictionary and push it to its corresponding buffer
                token, postings = next(iter(index.items()))
                buffers[fi].append((token, postings))

        if buffers[fi]:
            # heappush inserts an element while maintaining the minimum element at the top. O(logn) time complexity
            heapq.heappush(min_heap, (buffers[fi][0][0], fi))


    curr = None
    count = 0

    while min_heap:
        # heappop gets the smallest token from the min heap. However, it must rearrange the heap to get the next min. O(logn) time complexity
        token, fi = heapq.heappop(min_heap)

        # If it's a new token, push the old token to the output buffer since we've dealt with all occurences of it.
        if token != curr and curr is not None:
            offsets[curr] = output_file.tell()
            for doc in merged_index[curr]:
                merged_index[curr][doc]["fields"] = list(merged_index[curr][doc]["fields"])
            output_file.write(json.dumps({curr: merged_index[curr]}) + "\n")
            # Delete token from main memory
            del merged_index[curr]
            count += 1


        curr = token
        # Add new token to the set of unique tokens 
        unique_tokens.add(token) 
        t, postings = buffers[fi].pop(0)
        # Update the merged index in main memory
        for doc, data in postings.items():
            merged_index[token][doc]["freq"] += data["freq"]
            merged_index[token][doc]["fields"].update(data["fields"])


        # If the buffer for this file is empty, refill it if possible.
        if not buffers[fi]:
            buffers[fi].clear()
            for i in range(500):
                line = opened_files[fi].readline().strip()
                if not line:
                    break
                index = json.loads(line)
                if index:
                    token, postings = next(iter(index.items()))
                    buffers[fi].append((token, postings))

        if buffers[fi]:
            heapq.heappush(min_heap, (buffers[fi][0][0], fi))

        # Once we reach 500 tokens for this file, write the information to disk immediately
        if count >= 500:
            output_file.flush()
            count = 0

    # Makes sure the last token processed, if any, is written to disk
    if curr in merged_index:
        offsets[curr] = output_file.tell()
        for doc in merged_index[curr]:
            merged_index[curr][doc]["fields"] = list(merged_index[curr][doc]["fields"])
        output_file.write(json.dumps({curr: merged_index[curr]}) + "\n")

    # Close all files
    output_file.close()
    for f in opened_files.values():
        f.close()

    with open("INDEX/offset_positions.json", "w", encoding="utf-8") as f:
        json.dump(offsets, f)

    global urls
    with open("INDEX/url_map.json", "w", encoding="utf-8") as f:
        json.dump(urls, f)

    # Update the total unqiue tokens and size of final merged index file
    global total_unique_words, merged_index_size
    total_unique_words = len(unique_tokens)
    merged_index_size = os.path.getsize("INDEX/merged_index.json") / 1024




def print_info():
    log_write(" ")
    log_write("----------------###### STATISTICS ######----------------\n")
    
    log_write(f"The number of indexed documents: {total_documents}")
    log_write("\n")

    log_write(f"The number of unique words: {total_unique_words}")
    log_write("\n")


    log_write(f"The total size (in KB) of your index on disk: {merged_index_size}")
    log_write("\n")

    log_write(f"The number of partial indexes created: {total_partial_indexes}")
    log_write("\n")


    log_write("\n")
    log_write("----------------############################----------------")
    
    



def load_offset_positions():
    with open("INDEX/offset_positions.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_url_map():
    with open("INDEX/url_map.json", "r", encoding="utf-8") as f:
        return json.load(f)


def process_query(query):
    tokens = re.findall(r'[a-zA-Z0-9]+', query.lower())
    stemmed_tokens = set()
    for token in tokens:
        stemmed = porter_stemmer.stem(token)
        stemmed_tokens.add(stemmed)
    return stemmed_tokens


def search_query(query):
    query_tokens = process_query(query)
    if not query_tokens:
        return []

    offsets = load_offset_positions()
    list_of_sets = []

    with open("INDEX/merged_index.json", "r", encoding="utf-8") as f:
        for token in query_tokens:
            if token in offsets:
                f.seek(offsets[token]) 
                line = json.loads(f.readline().strip())
                posting = line.get(token, {})

                docs = set(posting.keys())
                list_of_sets.append(docs)

    if not list_of_sets:
        return []

    list_of_sets.sort(key=len)

    result = list_of_sets[0]
    for doc_set in list_of_sets[1:]:
        result &= doc_set

    return list(result)[:5]




def search_interface():
    log_write("\n--------SEARCH INTERFACE--------\n")
    url_map = load_url_map()

    while True:
        query = input("\nEnter your query (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        retrieved_docs = search_query(query)
        if retrieved_docs:
            for doc in retrieved_docs:
                url = url_map.get(doc, "Not Found")
                log_write(f"{url}")
        else:
            log_write("No relevant documents found")


    log_write("----------------------------------")



if __name__ == "__main__":
    create_partial_index()
    merge_indexes()
    print_info()
    search_interface()