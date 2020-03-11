import requests
import argparse
from requests.auth import HTTPBasicAuth
import base64
import os
import time


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, help='login name', required=True)
    parser.add_argument('--password', type=str, help='login password', required=True)
    parser.add_argument('--date', type=str, help='search date', required=True)
    parser.add_argument('--invoice_type_id', type=str, help='invoice type id', required=True)

    return parser.parse_args()


def get_tasks_of_page(date=None, invoice_type=None, next_url=None):
    api = next_url if next_url else \
        'http://52.193.30.103/argus/api/task/myte/?search={0}&invoice_type={1}'.format(date, invoice_type)

    r = requests.get(api, auth=auth, headers=headers)

    if r.ok:
        results = r.json().get('results')
        next = r.json().get('next')
        count = r.json().get('count')

        for result in results:
            global index
            index += 1

            print('\r download: {0}/{1}'.format(index, count), end='')
            get_image(result['id'])

    else:
        print('Download task info failed.')
        print('Call api: {url}'.format(url=api))
        print('[Err Reason]:{err}'.format(err=r.reason))
        print('[Err text]:{err}'.format(err=r.text))
        next = None

    return next


def get_image(task_id):
    api = 'http://52.193.30.103/argus/api/task_file/myte/{0}'.format(task_id)
    r = requests.get(api, auth=auth, headers=headers)
    if r.ok:
        result = r.json()[0]
        quality_file = result.get('quality_file')
        file_name = os.path.split(quality_file)[-1]

        with open(os.path.join(path, file_name), 'wb') as f:
            image_64_decode = base64.b64decode(result.get('picture').encode("utf8"))
            f.write(image_64_decode)

    else:
        print('Download image {id} failed.'.format(id=task_id))
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
    time_flag = int(time.time())
    for date in args.date.split(','):
        index = 0
        path = '{root}/invoice_type_{type}-{time}/{date}'.format(
            root=os.path.abspath('.'), date=date, time=time_flag, type=args.invoice_type_id
        )
        os.makedirs(path)
        print('\ndownload path: {0}'.format(path))

        next_url = get_tasks_of_page(date=date, invoice_type=args.invoice_type_id)
        while True:
            if next_url:
                next_url = get_tasks_of_page(next_url=next_url)
            else:
                break
