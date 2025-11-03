/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "cmn/path_base.h"
#include "space/space_property.h"
#include "space/space_layout.h"
#include "space/space_shard_layout.h"
#include "space/space_manager.h"

class UCSpacePropertyTest : public UC::PathBase {};
 
/*
* check the persistence of property
*/
TEST_F(UCSpacePropertyTest, CapacityPersistence)
{
   size_t blocksize = 1024 * 1024;
   UC::SpaceManager spaceMgr;
   ASSERT_EQ(spaceMgr.Setup({this->Path()}, blocksize, false, blocksize * 5), UC::Status::OK());
   const UC::SpaceLayout* layout = spaceMgr.GetSpaceLayout();

   const std::string path = layout->ClusterPropertyFilePath();

   UC::SpaceProperty spaceProperty;
   ASSERT_EQ(spaceProperty.Setup(path), UC::Status::OK());
   ASSERT_EQ(spaceProperty.GetCapacity(), 0);
   
   spaceProperty.IncreaseCapacity(blocksize * 2);
   ASSERT_EQ(spaceProperty.GetCapacity(), blocksize*2);
   
   UC::SpaceProperty spaceProperty2;
   ASSERT_EQ(spaceProperty2.Setup(path), UC::Status::OK());
   ASSERT_EQ(spaceProperty2.GetCapacity(), blocksize*2);

   spaceProperty2.DecreaseCapacity(blocksize);
   ASSERT_EQ(spaceProperty2.GetCapacity(), blocksize);
   
   UC::SpaceProperty spaceProperty3;
   ASSERT_EQ(spaceProperty3.Setup(path), UC::Status::OK());
   ASSERT_EQ(spaceProperty3.GetCapacity(), blocksize);
 }