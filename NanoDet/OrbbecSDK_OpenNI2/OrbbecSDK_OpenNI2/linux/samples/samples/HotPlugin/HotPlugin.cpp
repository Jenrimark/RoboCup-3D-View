/*****************************************************************************
*                                                                            *
*  OpenNI 2.x Alpha                                                          *
*  Copyright (C) 2012 PrimeSense Ltd.                                        *
*                                                                            *
*  This file is part of OpenNI.                                              *
*                                                                            *
*  Licensed under the Apache License, Version 2.0 (the "License");           *
*  you may not use this file except in compliance with the License.          *
*  You may obtain a copy of the License at                                   *
*                                                                            *
*      http://www.apache.org/licenses/LICENSE-2.0                            *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*****************************************************************************/
#include <stdio.h>
#include <OpenNI.h>

#include "OniSampleUtilities.h"

#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms

using namespace openni;

openni::Device device;
bool deviceConnected;
bool escDown;

void analyzeFrame(const VideoFrameRef& frame)
{
	DepthPixel* pDepth;
	RGB888Pixel* pColor;

	int middleIndex = (frame.getHeight() + 1)*frame.getWidth() / 2;

	switch (frame.getVideoMode().getPixelFormat())
	{
	case PIXEL_FORMAT_DEPTH_1_MM:
	case PIXEL_FORMAT_DEPTH_100_UM:
		pDepth = (DepthPixel*)frame.getData();
		printf("D-[%08llu] %8d\n", (long long)frame.getTimestamp(),
			pDepth[middleIndex]);
		break;
	case PIXEL_FORMAT_RGB888:
		pColor = (RGB888Pixel*)frame.getData();
		printf("C-[%08llu] 0x%02x%02x%02x\n", (long long)frame.getTimestamp(),
			pColor[middleIndex].r & 0xff,
			pColor[middleIndex].g & 0xff,
			pColor[middleIndex].b & 0xff);
		break;
	default:
		printf("Unknown format\n");
	}
}


void connectDevice()
{
	Status rc;
	rc = device.open(ANY_DEVICE);
	if (rc != STATUS_OK)
	{
		printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
		return;
	}

	VideoStream depth, color;
	if (device.getSensorInfo(SENSOR_DEPTH) != NULL)
	{
		rc = depth.create(device, SENSOR_DEPTH);
		if (rc == STATUS_OK)
		{
			rc = depth.start();
			if (rc != STATUS_OK)
			{
				printf("Couldn't start the color stream\n%s\n", OpenNI::getExtendedError());
			}
		}
		else
		{
			printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
		}
	}

	if (device.getSensorInfo(SENSOR_COLOR) != NULL)
	{
		rc = color.create(device, SENSOR_COLOR);
		if (rc == STATUS_OK)
		{
			const Array<VideoMode>& colorViodeModes = color.getSensorInfo().getSupportedVideoModes();
			for (int i = 0; i < colorViodeModes.getSize(); i++)
			{
				VideoMode videoMode = colorViodeModes[i];
				if (videoMode.getPixelFormat() == PIXEL_FORMAT_RGB888)
				{
					printf("Set video mode w %d,h %d, FPS %d, PIXEL_FORMAT_RGB888\n", videoMode.getResolutionX(), videoMode.getResolutionY(), videoMode.getFps());
					color.setVideoMode(videoMode);
					break;
				}

			}

			rc = color.start();
			if (rc != STATUS_OK)
			{
				printf("Couldn't start the color stream\n%s\n", OpenNI::getExtendedError());
			}
		}
		else
		{
			printf("Couldn't create color stream\n%s\n", OpenNI::getExtendedError());
		}
	}

	VideoFrameRef frame;
	VideoStream* streams[] = { &depth, &color };
	while (deviceConnected)
	{
		if (wasKeyboardHit() == 1)
		{
			escDown = true;
			break;
		}
		int readyStream = -1;
		rc = OpenNI::waitForAnyStream(streams, 2, &readyStream, SAMPLE_READ_WAIT_TIMEOUT);
		if (rc != STATUS_OK)
		{
			printf("Wait failed! (timeout is %d ms)\n%s\n", SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
			continue;
		}

		switch (readyStream)
		{
		case 0:
			// Depth
			depth.readFrame(&frame);
			break;
		case 1:
			// Color
			color.readFrame(&frame);
			break;
		default:
			printf("Unxpected stream\n");
		}

		analyzeFrame(frame);
	}

	depth.stop();
	color.stop();
	depth.destroy();
	color.destroy();
	device.close();
}


void closeDevice()
{
	device.close();
}


class OpenNIDeviceListener : public OpenNI::DeviceConnectedListener,
	public OpenNI::DeviceDisconnectedListener,
	public OpenNI::DeviceStateChangedListener
{
public:
	virtual void onDeviceStateChanged(const DeviceInfo* pInfo, DeviceState state)
	{
		printf("Device \"%s\" error state changed to %d\n", pInfo->getUri(), state);
	}

	virtual void onDeviceConnected(const DeviceInfo* pInfo)
	{
		printf("Device \"%s\" connected\n", pInfo->getUri());
		deviceConnected = true;
	}

	virtual void onDeviceDisconnected(const DeviceInfo* pInfo)
	{
		printf("Device \"%s\" disconnected\n", pInfo->getUri());

		if (pInfo->getUsbProductId() == device.getDeviceInfo().getUsbProductId())
		{
			deviceConnected = false;
		}

	}
};


int main()
{
	escDown = false;
	Status rc = OpenNI::initialize();
	if (rc != STATUS_OK)
	{
		printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
		return 1;
	}

	/*Create device connect listener*/
	OpenNIDeviceListener devicePrinter;
	OpenNI::addDeviceConnectedListener(&devicePrinter);
	OpenNI::addDeviceDisconnectedListener(&devicePrinter);
	OpenNI::addDeviceStateChangedListener(&devicePrinter);

	openni::Array<openni::DeviceInfo> deviceList;
	openni::OpenNI::enumerateDevices(&deviceList);
	for (int i = 0; i < deviceList.getSize(); ++i)
	{
		printf("Device \"%s\" already connected\n", deviceList[i].getUri());
	}

	if (deviceList.getSize() > 0)
	{
		deviceConnected = true;
	}

	while (!wasKeyboardHit())
	{
		if (deviceConnected)
		{
			connectDevice();
		}

		if (escDown == true)
		{
			break;
		}
	}

	OpenNI::shutdown();

	return 0;
}
