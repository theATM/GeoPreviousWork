# Chat gpt-3 ai created code - with some prompt enginering
# To replace unused information from the xml annotations
# Leave only filename, size, segmented, and object (all the bboxes)
# Used with rsd-god original training set
import os
import shutil
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def copy_xml_files(xml_dir, xml_copy_dir):
    # Create the copy directory if it doesn't exist
    if not os.path.exists(xml_copy_dir):
        os.makedirs(xml_copy_dir)

    # Get the list of XML files in the directory
    xml_files = [file for file in os.listdir(xml_dir) if file.endswith('.xml')]

    for xml_file in xml_files:
        # Parse the XML file
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()

        # Create a new root element for the copy
        copy_root = ET.Element("annotation")

        # Copy specific children of the original root to the copy root
        for child in root:
            if child.tag in ["filename", "size", "segmented", "object"]:
                copy_root.append(child)

        # Create a new XML tree with the copy root
        copy_tree = ET.ElementTree(copy_root)
        ET.indent(tree, '  ')

        # Generate a pretty XML string from the copy root
        copy_file_path = os.path.join(xml_copy_dir, xml_file)
        copy_tree_str = ET.tostring(copy_root, encoding="utf-8")
        dom = minidom.parseString(copy_tree_str)
        pretty_xml = dom.toprettyxml(indent="  ", newl="")

        # Remove the <?xml version="1.0" ?> declaration
        pretty_xml = pretty_xml.replace('<?xml version="1.0" ?>', '')
        
        # Add a newline character after the opening <annotation> tag
        pretty_xml = pretty_xml.replace('<annotation>', '<annotation>\n  ', 1)

        # Parse the pretty XML string back into an ElementTree
        copy_tree = ET.ElementTree(ET.fromstring(pretty_xml))

        #Save the copy XML file
        copy_tree.write(copy_file_path)

    print("XML files copied successfully!")

# Provide the directory paths for XML files and the copy directory
xml_dir = "xml"
xml_copy_dir = "xml_copy"

copy_xml_files(xml_dir, xml_copy_dir)

