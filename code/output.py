import logging
import os
from xml.dom import minidom

import utils
import xml.etree.ElementTree as etree


def save_as_xml(document, filename):
    doc_id = document.id
    data_root = etree.Element("data")
    root = etree.SubElement(data_root, "annotations")
    for entity in document.get_entities():
        ent = etree.SubElement(root, "entity")

        id = etree.SubElement(ent, "id")
        id.text = "{}@{}@{}".format(str(entity.id), 'e', doc_id)

        span = etree.SubElement(ent, "span")
        span.text = "{},{}".format(entity.span[0], entity.span[1])

        type = etree.SubElement(ent, "type")
        type.text = entity.get_class().upper()

        properties = etree.SubElement(ent, "properties")

        if entity.get_class().startswith("E"):
            dct = etree.SubElement(properties, "DocTimeRel")
            dct.text = entity.doc_time_rel
            # maybe more...

    for relation in document.get_relations():
        ent = etree.SubElement(root, "relation")

        id = etree.SubElement(ent, "id")
        id.text = "{}@{}@{}".format(str(relation.id), 'r', doc_id)

        type = etree.SubElement(ent, "type")
        type.text = "TLINK"

        properties = etree.SubElement(ent, "properties")

        source = etree.SubElement(properties, "Source")
        source.text = "{}@{}@{}".format(str(relation.source.id), 'e', doc_id)

        type = etree.SubElement(properties, "Type")
        type.text = relation.class_type

        target = etree.SubElement(properties, "Target")
        target.text = "{}@{}@{}".format(str(relation.target.id), 'e', doc_id)

    with open(filename, 'w') as f:
        f.write(prettify(data_root))
    logging.info("outputted " + filename + "!")


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = etree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def output_doc(document, outputpath=utils.outputpath):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    newpath = os.path.join(outputpath, document.id)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    xml_path = os.path.join(newpath, document.id + "Temporal.xml")
    save_as_xml(document, xml_path)


def output_docs_as_xml(documents, outputpath=utils.outputpath):
    for document in documents:
        output_doc(document, outputpath)


if __name__ == "__main__":
    docs = utils.get_documents_from_file(utils.store_path)
    for document in docs:
        newpath = os.path.join(utils.outputpath, document.id)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        xml_path = os.path.join(newpath, document.id + ".xml")
        save_as_xml(document, xml_path)
