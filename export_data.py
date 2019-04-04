import requests
import argparse
from requests.auth import HTTPBasicAuth
import sys
import json
import os
import time


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, help='login name', required=True)
    parser.add_argument('--password', type=str, help='login password', required=True)
    parser.add_argument('--server', type=str, help='Server path', required=True)
    parser.add_argument('--project', type=str, help='Project name', required=True)
    parser.add_argument('--task', type=int, help='Task id', required=True)
    parser.add_argument('--label', dest='label', help='Download label information', action='store_true')
    parser.add_argument('--postprocessing', dest='postprocessing', help='Download postprocessing information', action='store_true')
    parser.add_argument('--source_file', dest='source_file', help='Download source file', action='store_true')
    parser.add_argument('--quality_file', dest='quality_file', help='Download quality file', action='store_true')
    parser.add_argument('--engine_raw_result', dest='engine_raw_result', help='Download engine raw result', action='store_true')
    parser.add_argument('--engine_result', dest='engine_result', help='Download engine result', action='store_true')
    parser.add_argument('--only_labelled_data', dest='only_labelled_data', action='store_true', default=None)
    parser.add_argument('--only_unlabelled_data', dest='only_unlabelled_data', action='store_true', default=None)
    parser.add_argument('--only_valid_data', dest='only_valid_data', action='store_true', default=None)
    parser.add_argument('--only_invalid_data', dest='only_invalid_data', action='store_true', default=None)
    return parser.parse_args()


def get_engine(api):
    r = requests.get(api, auth=auth, headers=headers)
    if r.ok:
        return json.loads(r.json().get('json_result', '{}')).get('text_boxes')
    else:
        print('Download ticket {id} engine info failed.'.format(id=ticket_id))
        print('Call api: {url}'.format(url=api))
        print('[Err Reason]:{err}'.format(err=r.reason))
        print('[Err text]:{err}'.format(err=r.text))
        return None


def download_info(server, project, task_id):
    params = dict()
    if args.only_labelled_data is True:
        params['only_label_info'] = True

    if args.only_unlabelled_data is True:
        params['only_label_info'] = False

    if args.only_valid_data is True:
        params['ticket_invalid'] = False

    if args.only_invalid_data is True:
        params['ticket_invalid'] = True

    label_api = '{server}/argus/api/{project}/task_info/{id}/'.format(
        server=server, project=project, id=task_id
    )
    r = requests.get(label_api, auth=auth, headers=headers, params=params)
    if r.ok:
        return r.json()
    else:
        print('Download task info failed.')
        print('Call api: {url}'.format(url=label_api))
        print('[Err Reason]:{err}'.format(err=r.reason))
        print('[Err text]:{err}'.format(err=r.text))
        sys.exit(1)


def download_file(output_path, key, label):
    os.mkdir(output_path)
    for ticket in label:
        api = label[ticket][key]
        r = requests.get(api, auth=auth, headers=headers)
        if r.ok:
            r = requests.get(api, allow_redirects=True, auth=auth, stream=True)
            source_name = r.headers['Content-Disposition'].split('=')[-1]
            rename_source = '{0}{1}'.format(ticket, os.path.splitext(source_name)[-1])
            with open(os.path.join(output_path, rename_source), 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)

            print('download: {file}'.format(file=rename_source))
        else:
            print('Download file failed.')
            print('Call api: {url}'.format(url=api))
            print('[Err Reason]:{err}'.format(err=r.reason))
            print('[Err text]:{err}'.format(err=r.text))


def download_postprocessing_result(task_info):
    dir_name = 'task_{id}_algorithm_results_{time}'.format(id=args.task, time=int(time.time()))
    os.mkdir(dir_name)

    for ticket_id in task_info:
        api = task_info[ticket_id]['postprocessing_result']
        r = requests.get(api, auth=auth, headers=headers)
        try:
            if r.ok:
                json_result = r.json().get('json_result')
                if json_result is None:
                    continue

                with open('{path}/{id}.json'.format(path=dir_name, id=ticket_id), 'w') as f:
                    f.write(json_result)

                print('Download: {file}.json'.format(file=ticket_id))
            else:
                print('Download file failed.')
                print('Call api: {url}'.format(url=api))
                print('[Err Reason]:{err}'.format(err=r.reason))
                print('[Err text]:{err}'.format(err=r.text))

        except Exception as e:
            print('Download file failed.')
            print('Call api: {url}'.format(url=api))
            print('[Err]:{err}'.format(err=str(e)))


def download_label_info(task_info):
    dir_name = 'task_{id}_label_{time}'.format(id=args.task, time=int(time.time()))
    result = dict(fields=[], text_boxes=[], invoice_type=None)
    os.mkdir(dir_name)

    for ticket_id in task_info:
        api = task_info[ticket_id]['label_result']
        if api is None:
            continue

        r = requests.get(api, auth=auth, headers=headers)
        if r.ok:
            try:
                result['invoice_type'] = r.json().get('ticket_type')
                result['ticket'] = r.json().get('ticket')
                result['country'] = r.json().get('country')
                result['fields'] = json.loads(r.json().get('labeling')).get('fields', [])
                result['tables'] = json.loads(r.json().get('labeling')).get('tables', [])
            except Exception as e:
                print('download label {file}.json failed'.format(file=ticket_id))
                print(str(e))
                continue

            result['text_boxes'] = get_engine(
                task_info[ticket_id]['task_result_api']
            )

            with open('{path}/{id}.json'.format(path=dir_name, id=ticket_id), 'w') as f:
                f.write(json.dumps(result))

            print('download: {file}.json'.format(file=ticket_id))
        else:
            print('Download file failed.')
            print('Call api: {url}'.format(url=api))
            print('[Err Reason]:{err}'.format(err=r.reason))
            print('[Err text]:{err}'.format(err=r.text))


if __name__ == "__main__":
    args = init_args()
    auth = HTTPBasicAuth(args.user, args.password)
    headers = {
        'Content-Type': "application/json",
        'Cache-Control': "no-cache",
    }

    # print('==> Download start')
    task_info = download_info(args.server, args.project, args.task)

    if args.label is True:
        print('==> Download label info start...')
        download_label_info(task_info)
        print('==> Download label info done.')

    if args.postprocessing is True:
        print('==> Download postprocessing info start...')
        download_postprocessing_result(task_info)
        print('==> Download postprocessing info done.')

    if args.source_file is True:
        print('==> Download source file start...')
        output_path = 'task_{id}_source_file_{time}'.format(id=args.task, time=int(time.time()))
        download_file(output_path, 'source_file_path', task_info)
        print('==> Download source file done.')

    if args.quality_file is True:
        print('==> Download quality file start...')
        output_path = 'task_{id}_quality_file_{time}'.format(id=args.task, time=int(time.time()))
        download_file(output_path, 'quality_file_path', task_info)
        print('==> Download quality file done.')

    if args.engine_raw_result is True:
        print('==> Download engine raw file start...')
        output_path = 'task_{id}_engine_raw_file_{time}'.format(id=args.task, time=int(time.time()))
        download_file(output_path, 'engine_raw_result_path', task_info)
        print('==> Download engine raw file done.')

    if args.engine_result is True:
        print('==> Download engine file start...')
        output_path = 'task_{id}_engine_file_{time}'.format(id=args.task, time=int(time.time()))
        download_file(output_path, 'engine_result_path', task_info)
        print('==> Download engine file done.')
