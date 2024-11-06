#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retrives inpections from pre-existent mongo collection and simulate
the updating of the 'gscs_classification' mongo collection
"""

__author__ = "Domingos Rodrigues"
__email__ = "domingos.rodrigues@inventvision.com.br"
__copyright__ = "Copyright (C) 2024 Invent Vision"
__license__ = "Strictly proprietary for Invent Vision."

import argparse
import random
import time
from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta
from logzero import logger
from pymongo import ASCENDING

from utils import connect_to_mongo, scrapRank


def sample_from_dict(data_dict, sample_size=20):
    sampled_data = {}
    for key, value_list in data_dict.items():
        # Sample the items, making sure not to exceed the list length
        sampled_data[key] = random.sample(value_list, min(sample_size, len(value_list)))
    return sampled_data


def getRandomInspectionsFromEachClass(
    mongo_db,
    count=20,
    beginDate=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    endDate=datetime.now(),
):
    if not mongo_db:
        raise Exception("Database not connected!")

    indexes = mongo_db.inspections.index_information()
    if "date_1" not in indexes:
        index_name = mongo_db.inspections.create_index([("date", ASCENDING)])
        logger.info(f"Created index '{index_name}' on 'date' field")

    # avoid inspections with no classification
    pipeline = [
        {
            "$match": {
                "date": {
                    "$gte": beginDate,
                    "$lt": endDate,
                },
                "result.detection": {"$ne": None},
            }
        },
        {
            "$group": {
                "_id": "$result.detection.class",
                "documents": {"$push": "$$ROOT"},
            }
        },
        {"$unwind": {"path": "$documents"}},
    ]

    # logger.info(f"pipeline = {pipeline}")
    # explain_output = mongo_db.command('aggregate', 'inspections', pipeline=pipeline, explain=True)
    # logger.info(explain_output)

    inspections_res = mongo_db.inspections.aggregate(pipeline)

    inspections_per_class = {}
    for group in inspections_res:
        class_name = group["_id"]
        if class_name in inspections_per_class:
            inspections_per_class[class_name].append(group["documents"])
        else:
            inspections_per_class[class_name] = []
    inspections_per_class.pop("no_aplica", None)

    new_sample = sample_from_dict(inspections_per_class, sample_size=count)
    for k, v in new_sample.items():
        logger.info(f"{k}: {len(v)}")
    return new_sample


def saveIntegrationRequest(parGscsId, mongo_db):
    data = {
        "gscs_id": parGscsId,
        "need_sync": True,
        "retry_sync_count": 0,
        "inspector": None,
        "grade": None,
        "cls_manual": None,
    }
    mongo_db.gscs_classifications.insert_one(data)


def inject_inspections(sample, mongo_db, drop=False):
    """_summary_
    Assume that the collection gscs_classifications does not exist

    Arguments:
        sample -- _description_
        db_destination -- _description_
    """
    if not mongo_db:
        raise Exception("Database not connected!")

    # Check if the collection exists
    for collection in ["inspections", "gscs_classifications"]:
        if drop:
            if collection in mongo_db.list_collection_names():
                print(f"Collection '{collection}' exists. Dropping it.")
                mongo_db[collection].drop()  # Drop the existing collection

        if collection not in mongo_db.list_collection_names():
            mongo_db.create_collection(collection)

    # insert_many() should be more performant
    # max_id = mongo_db.inspections.find_one().sort("gscs_id", -1)
    max_id = mongo_db.inspections.find_one(sort=[("gscs_id", -1)])
    parGscsId = max_id["gscs_id"] + 1 if max_id else 1000000
    for class_name, insp_list in sample.items():
        for insp in insp_list:
            logger.info(f"inject inspection: {insp['_id']}")
            insp["gscs_id"] = parGscsId
            res = mongo_db.inspections.insert_one(insp)
            if res.acknowledged:
                saveIntegrationRequest(parGscsId, mongo_db)
            else:
                logger.error(f"Inspection {insp['_id']} could not be inserted")
            parGscsId += 1


def getPendingClassifications(mongo_db, force=False):
    if force:
        pending = list(mongo_db.gscs_classifications.find())
    else:
        query = {"need_sync": True}
        pending = list(mongo_db.gscs_classifications.find(query))
    return pending


def updateGSCSClassification(gscs_id, gscs_info, mongo_db):
    update_data = {
        "$set": {
            "grade": gscs_info["grade"],
            "cls_manual": gscs_info["cls_manual"],
            "inspector": gscs_info["inspector"],
            "need_sync": False,
        },
        "$inc": {"retry_sync_count": 1},
    }
    mongo_db.gscs_classifications.update_one({"gscs_id": gscs_id}, update_data)


def update_inspections(mongo_db, force=False):
    if not mongo_db:
        raise Exception("Database not connected!")

    pending_cls = getPendingClassifications(mongo_db, force=force)

    for cls in pending_cls:
        gscs_id = cls["gscs_id"]
        # retrieve associated inspection
        query = {"gscs_id": gscs_id}
        insp = mongo_db.inspections.find_one(query)

        if insp is None:
            logger.error(f"Inspection with gscs_id '{gscs_id}' not found")
        else:
            # retrieve AI class code
            insp_class_code = insp["result"]["detection"]["classCode"]
            insp_class_name = scrapRank[insp_class_code][1]
            logger.info(f"insp {insp['_id']} => {insp_class_code} [{insp_class_name}]")

            # Assume that Human and AI classifications are the same
            gscs_info = {
                "grade": insp_class_code,
                "cls_manual": insp_class_name,
                "inspector": r"IVISION\EMPLOYEE",
            }
            # logger.info(f"gscs_info[{gscs_id}] = {gscs_info}")

            updateGSCSClassification(gscs_id, gscs_info, mongo_db)
        time.sleep(3)


def get_date_range(collection):
    pipeline = [
        {
            "$group": {
                "_id": None,
                "min_date": {
                    "$min": "$date"
                },  # Replace 'date' with your date field name
                "max_date": {"$max": "$date"},
            }
        }
    ]
    result = list(collection.aggregate(pipeline))
    if result:
        return result[0]["min_date"], result[0]["max_date"]
    else:
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Connects to MongoDB source databases (from Sahagun or Tultitlán sites), retrieves
a representative sample of inspections, and injects them into a target MongoDB
database instance for data analysis.
This script is designed to facilitate the creation and management of a test MongoDB
database by sampling inspection data from production databases located at the Sahagun
and Tultitlan sites.
""",
        add_help=True,
    )

    parser.add_argument(
        "--mongo_url_source",
        type=str,
        required=False,
        default="127.0.0.1:27018",
        help="MongoDB SOURCE url (Sahagún=>27018; Tultitlán=>27019) [default: %(default)s]",
    )

    parser.add_argument(
        "--mongo_url_dest",
        type=str,
        required=False,
        default="127.0.0.1:27017",
        help="MongoDB test DESTINATION url [default: %(default)s]",
    )

    parser.add_argument(
        "--count",
        type=int,
        required=False,
        default=10,
        help="Number of inspections per class to retrieve and migrate [defaul: %(default)s]",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="commands")

    # injecting data subcommend
    inject_parser = subparsers.add_parser(
        "inject",
        help="Migrate data from MongoDB source databases to the destination test database",
    )
    inject_parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop destination collections before data migration",
    )

    # Update subcommand
    update_parser = subparsers.add_parser(
        "update", help="Update GSCS database with new GSCS info for each inspection"
    )
    update_parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh of GSCS info for inspections (even if attribute 'need_sync' of inspection is False)",
    )

    args = parser.parse_args()

    mongo_dest_host, mongo_dest_port = args.mongo_url_dest.split(":")
    db_destination = connect_to_mongo(mongo_dest_host, int(mongo_dest_port))

    if args.command in ["inject"]:
        logger.info(f"Start collecting samples from {args.mongo_url_source}")
        mongo_src_host, mongo_src_port = args.mongo_url_source.split(":")
        mongo_db_source = connect_to_mongo(mongo_src_host, int(mongo_src_port))
        if args.drop:
            # Drop and repopulate the collection with initial data
            samples_dict = getRandomInspectionsFromEachClass(
                mongo_db_source, count=args.count
            )
        else:
            # add sample inspections to the existing collection, avoiding duplicate entries

            start_date, end_date = get_date_range(db_destination.inspections)
            if all([start_date, end_date]):
                # change date interval to avoid conflicts (duplicated inspections)
                end_date = start_date
                start_date -= relativedelta(months=6)
                print(f"New Date range for new sample: {start_date} to {end_date}")
                samples_dict = getRandomInspectionsFromEachClass(
                    mongo_db_source,
                    count=args.count,
                    beginDate=start_date,
                    endDate=end_date,
                )
            else:
                # default
                samples_dict = getRandomInspectionsFromEachClass(
                    mongo_db_source, count=args.count
                )

        logger.info(f"Migrate sample to {args.mongo_url_dest}")
        inject_inspections(samples_dict, db_destination, drop=args.drop)

    if args.command in ["update"]:
        logger.info("Simulate availability of GSCS human classifications")
        logger.info("Update the inspections with new GSCS info")

        update_inspections(db_destination, force=True)