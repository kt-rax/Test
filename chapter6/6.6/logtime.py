# -*- coding: utf-8 -*-
import logging

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}

for i in range(1,10):
    logger = logging.getLogger('位置')
    logger.warning('' ,extra=d)

