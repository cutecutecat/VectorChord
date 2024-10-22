#!/usr/bin/env bash
set -e

printf "SEMVER = ${SEMVER}\n"
printf "VERSION = ${VERSION}\n"
printf "ARCH = ${ARCH}\n"
printf "PLATFORM = ${PLATFORM}\n"

rm -rf ./build/dir_zip
rm -rf ./build/rabbithole-pg${VERSION}_${ARCH}-unknown-linux-gnu_${SEMVER}.zip
rm -rf ./build/dir_deb
rm -rf ./build/rabbithole-pg${VERSION}_${SEMVER}_${PLATFORM}.deb

mkdir -p ./build/dir_zip
cp ./target/schema.sql ./build/dir_zip/rabbithole--$SEMVER.sql
sed -e "s/@CARGO_VERSION@/$SEMVER/g" < ./rabbithole.control > ./build/dir_zip/rabbithole.control
cp ./target/${ARCH}-unknown-linux-gnu/release/librabbithole.so ./build/dir_zip/rabbithole.so
zip ./build/rabbithole-pg${VERSION}_${ARCH}-unknown-linux-gnu_${SEMVER}.zip -j ./build/dir_zip/*

mkdir -p ./build/dir_deb
mkdir -p ./build/dir_deb/DEBIAN
mkdir -p ./build/dir_deb/usr/share/postgresql/$VERSION/extension/
mkdir -p ./build/dir_deb/usr/lib/postgresql/$VERSION/lib/
for file in $(ls ./build/dir_zip/*.sql | xargs -n 1 basename); do
    cp ./build/dir_zip/$file ./build/dir_deb/usr/share/postgresql/$VERSION/extension/$file
done
for file in $(ls ./build/dir_zip/*.control | xargs -n 1 basename); do
    cp ./build/dir_zip/$file ./build/dir_deb/usr/share/postgresql/$VERSION/extension/$file
done
for file in $(ls ./build/dir_zip/*.so | xargs -n 1 basename); do
    cp ./build/dir_zip/$file ./build/dir_deb/usr/lib/postgresql/$VERSION/lib/$file
done
echo "Package: rabbithole-pg${VERSION}
Version: ${SEMVER}
Section: database
Priority: optional
Architecture: ${PLATFORM}
Maintainer: Tensorchord <support@tensorchord.ai>
Description: Vector database plugin for Postgres, written in Rust, specifically designed for LLM
Homepage: https://pgvecto.rs/
License: apache2" \
> ./build/dir_deb/DEBIAN/control
(cd ./build/dir_deb && md5sum usr/share/postgresql/$VERSION/extension/* usr/lib/postgresql/$VERSION/lib/*) > ./build/dir_deb/DEBIAN/md5sums
dpkg-deb -Zxz --build ./build/dir_deb/ ./build/rabbithole-pg${VERSION}_${SEMVER}_${PLATFORM}.deb